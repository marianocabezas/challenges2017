from keras import backend as K
from keras.layers import Dense, Conv3D, Dropout, Flatten, Input, concatenate, Reshape, Lambda
from keras.layers import BatchNormalization, LSTM, Permute, Activation, PReLU, Average
from keras.models import Model
from itertools import product
import numpy as np


def compile_network(inputs, outputs, weights):
    net = Model(inputs=inputs, outputs=outputs)

    net.compile(
        optimizer='adadelta',
        loss='categorical_crossentropy',
        loss_weights=weights,
        metrics=['accuracy']
    )

    return net


def get_convolutional_block(input_l, filters_list, kernel_size_list, activation=PReLU, drop=0.5):
    for filters, kernel_size in zip(filters_list, kernel_size_list):
        input_l = Conv3D(filters, kernel_size=kernel_size, data_format='channels_first')(input_l)
        # input_l = BatchNormalization(axis=1)(input_l)
        input_l = activation()(input_l)
        input_l = Dropout(drop)(input_l)

    return input_l


def get_tissue_binary_stuff(input_l):
    csf = Dense(2)(input_l)
    gm = Dense(2)(input_l)
    wm = Dense(2)(input_l)
    csf_out = Activation('softmax', name='csf')(csf)
    gm_out = Activation('softmax', name='gm')(gm)
    wm_out = Activation('softmax', name='wm')(wm)

    return csf, gm, wm, csf_out, gm_out, wm_out


def get_iseg_baseline(input_shape, filters_list, kernel_size_list, dense_size):
    merged_inputs = Input(shape=input_shape, name='merged_inputs')
    # Input splitting
    input_shape = K.int_shape(merged_inputs)
    t1 = Lambda(lambda l: K.expand_dims(l[:, 0, :, :, :], axis=1), output_shape=(1,) + input_shape[2:])(merged_inputs)
    t2 = Lambda(lambda l: K.expand_dims(l[:, 1, :, :, :], axis=1), output_shape=(1,) + input_shape[2:])(merged_inputs)

    # Convolutional part
    t2 = get_convolutional_block(t2, filters_list, kernel_size_list)
    t1 = get_convolutional_block(t1, filters_list, kernel_size_list)

    # Tissue binary stuff
    t2_f = Flatten()(t2)
    t1_f = Flatten()(t1)
    t2_f = Dense(dense_size, activation='relu')(t2_f)
    t2_f = Dropout(0.5)(t2_f)
    t1_f = Dense(dense_size, activation='relu')(t1_f)
    t1_f = Dropout(0.5)(t1_f)
    merged = concatenate([t2_f, t1_f])
    csf, gm, wm, csf_out, gm_out, wm_out = get_tissue_binary_stuff(merged)

    # Final labeling
    merged = concatenate([t2_f, t1_f, PReLU()(csf), PReLU()(gm), PReLU()(wm)])
    merged = Dropout(0.5)(merged)
    brain = Dense(4, name='brain', activation='softmax')(merged)

    # Weights and outputs
    weights = [0.2,     0.5,    0.5,    1.0]
    outputs = [csf_out, gm_out, wm_out, brain]

    return compile_network(merged_inputs, outputs, weights)


def get_iseg_experimental1(input_shape, filters_list, kernel_size_list, dense_size):
    merged_inputs = Input(shape=input_shape, name='merged_inputs')
    # Convolutional stuff
    merged = get_convolutional_block(merged_inputs, filters_list, kernel_size_list)

    # Tissue binary stuff
    merged_f = Flatten()(merged)
    merged_f = Dense(dense_size, activation='relu')(merged_f)
    merged_f = Dropout(0.5)(merged_f)
    csf, gm, wm, csf_out, gm_out, wm_out = get_tissue_binary_stuff(merged_f)

    # Final labeling stuff
    merged = concatenate([PReLU()(csf), PReLU()(gm), PReLU()(wm), merged_f])
    merged = Dropout(0.5)(merged)
    brain = Dense(4, activation='softmax', name='brain')(merged)

    # Weights and outputs
    weights = [0.2,     0.5,    0.5,    1.0]
    outputs = [csf_out, gm_out, wm_out, brain]

    return compile_network(merged_inputs, outputs, weights)


def get_iseg_experimental2(input_shape, filters_list, kernel_size_list, dense_size):
    merged_inputs = Input(shape=input_shape, name='merged_inputs')
    # Convolutional part
    merged = get_convolutional_block(merged_inputs, filters_list, kernel_size_list)

    # LSTM stuff
    patch_center = Reshape((filters_list[-1], -1))(merged)
    patch_center = Dense(4, name='pre_rf')(Permute((2, 1))(patch_center))
    rf = LSTM(4, implementation=1)(patch_center)
    rf_out = Activation('softmax', name='rf_out')(rf)
    rf = Dropout(0.5)(PReLU(name='rf')(rf))

    # Tissue binary stuff
    merged_f = Flatten()(merged)
    merged_f = Dense(dense_size, activation='relu')(merged_f)
    merged_f = Dropout(0.5)(merged_f)
    csf, gm, wm, csf_out, gm_out, wm_out = get_tissue_binary_stuff(merged_f)

    # Brain labeling
    csf = Dropout(0.5)(PReLU()(csf))
    gm = Dropout(0.5)(PReLU()(gm))
    wm = Dropout(0.5)(PReLU()(wm))
    merged = concatenate([csf, gm, wm, merged_f])
    merged = Dropout(0.5)(merged)
    brain = Dense(4)(merged)
    br_out = Activation('softmax', name='brain_out')(brain)
    brain = Dropout(0.5)(PReLU(name='brain')(brain))

    # Final labeling
    final_layers = concatenate([csf, gm, wm, brain, rf])
    final = Dense(4, name='merge', activation='softmax')(final_layers)

    # Weights and outputs
    weights = [0.2,     0.5,    0.5,    0.8,    0.8,    1.0]
    outputs = [csf_out, gm_out, wm_out, br_out, rf_out, final]

    return compile_network(merged_inputs, outputs, weights)


def get_iseg_experimental3(input_shape, filters_list, kernel_size_list, dense_size):
    merged_inputs = Input(shape=input_shape, name='merged_inputs')
    # Input splitting
    input_shape = K.int_shape(merged_inputs)
    t1 = Lambda(lambda l: K.expand_dims(l[:, 0, :, :, :], axis=1), output_shape=(1,) + input_shape[2:])(merged_inputs)
    t2 = Lambda(lambda l: K.expand_dims(l[:, 1, :, :, :], axis=1), output_shape=(1,) + input_shape[2:])(merged_inputs)

    # Convolutional part
    t2 = get_convolutional_block(t2, filters_list, kernel_size_list)
    t1 = get_convolutional_block(t1, filters_list, kernel_size_list)

    # Tissue binary stuff
    t2_f = Flatten()(t2)
    t1_f = Flatten()(t1)
    t2_f = Dense(dense_size, activation='relu')(t2_f)
    t2_f = Dropout(0.5)(t2_f)
    t1_f = Dense(dense_size, activation='relu')(t1_f)
    t1_f = Dropout(0.5)(t1_f)
    merged = concatenate([t2_f, t1_f])
    csf, gm, wm, csf_out, gm_out, wm_out = get_tissue_binary_stuff(merged)

    full = Conv3D(dense_size, kernel_size=(1, 1, 1), data_format='channels_first')(concatenate([t1, t2], axis=1))
    full = PReLU()(full)
    full = Conv3D(dense_size/2, kernel_size=(1, 1, 1), data_format='channels_first')(full)
    full = PReLU()(full)
    full = Conv3D(dense_size/4, kernel_size=(1, 1, 1), data_format='channels_first')(full)
    full = PReLU()(full)
    full = Conv3D(4, kernel_size=(1, 1, 1), data_format='channels_first')(full)

    full_shape = K.int_shape(full)

    # x LSTM
    x_combos = product(range(full_shape[-2]), range(full_shape[-1]))
    lambda_x = Lambda(lambda l: l[:, :, :, i, j], output_shape=(4, full_shape[-3]))
    lambda_x_rev = Lambda(lambda l: l[:, :, -1::-1, i, j], output_shape=(4, full_shape[-3]))
    x_input = [lambda_x(PReLU()(full)) for (i, j) in x_combos] +[lambda_x_rev(PReLU()(full)) for (i, j) in x_combos]
    x_lstm = [LSTM(4, implementation=1)(x) for x in x_input]

    # y LSTM
    y_combos = product(range(full_shape[-3]), range(full_shape[-1]))
    lambda_y = Lambda(lambda l: l[:, :, i, :, j], output_shape=(4, full_shape[-2]))
    lambda_y_rev = Lambda(lambda l: l[:, :, i, -1::-1, j], output_shape=(4, full_shape[-2]))
    y_input = [lambda_y(PReLU()(full)) for (i, j) in y_combos] + [lambda_y_rev(PReLU()(full)) for (i, j) in y_combos]
    y_lstm = [LSTM(4, implementation=1)(y) for y in y_input]

    # z LSTM
    z_combos = product(range(full_shape[-3]), range(full_shape[-2]))
    lambda_z = Lambda(lambda l: l[:, :, i, j, :], output_shape=(4, full_shape[-1]))
    lambda_z_rev = Lambda(lambda l: l[:, :, i, j, -1::-1], output_shape=(4, full_shape[-1]))
    z_input = [lambda_z(PReLU()(full)) for (i, j) in z_combos] + [lambda_z_rev(PReLU()(full)) for (i, j) in z_combos]
    z_lstm = [LSTM(4, implementation=1)(PReLU()(z)) for z in z_input]

    # Final LSTM
    rf = Average()(x_lstm + y_lstm + z_lstm)

    # FC labeling
    full = Reshape((4, -1))(full)
    full = Permute((2, 1))(full)
    full_out = Activation('softmax', name='fc_out')(full)
    # rf = LSTM(4, implementation=1)(Reshape((4, -1))(full))
    # rf = Dense(4)(Flatten()(full))

    # Final labeling
    merged = concatenate([t2_f, t1_f, PReLU()(csf), PReLU()(gm), PReLU()(wm), PReLU()(rf)])
    # merged = concatenate([t2_f, t1_f, PReLU()(csf), PReLU()(gm), PReLU()(wm), PReLU()(Flatten()(full))])
    merged = Dropout(0.5)(merged)
    brain = Dense(4, name='brain', activation='softmax')(merged)

    # Weights and outputs
    weights = [0.2,     0.5,    0.5,    1.0,   0.8]
    outputs = [csf_out, gm_out, wm_out, brain, full_out]

    return compile_network(merged_inputs, outputs, weights)


def get_iseg_experimental4(input_shape, filters_list, kernel_size_list, dense_size):
    merged_inputs = Input(shape=input_shape, name='merged_inputs')
    # Input splitting
    input_shape = K.int_shape(merged_inputs)
    t1 = Lambda(lambda l: K.expand_dims(l[:, 0, :, :, :], axis=1), output_shape=(1,) + input_shape[2:])(merged_inputs)
    t2 = Lambda(lambda l: K.expand_dims(l[:, 1, :, :, :], axis=1), output_shape=(1,) + input_shape[2:])(merged_inputs)

    # Convolutional part
    t2 = get_convolutional_block(t2, filters_list, kernel_size_list)
    t1 = get_convolutional_block(t1, filters_list, kernel_size_list)

    # Tissue binary stuff
    t2_f = Flatten()(t2)
    t1_f = Flatten()(t1)
    t2_f = Dense(dense_size, activation='relu')(t2_f)
    t2_f = Dropout(0.5)(t2_f)
    t1_f = Dense(dense_size, activation='relu')(t1_f)
    t1_f = Dropout(0.5)(t1_f)
    merged = concatenate([t2_f, t1_f])
    csf, gm, wm, csf_out, gm_out, wm_out = get_tissue_binary_stuff(merged)

    full = Conv3D(dense_size, kernel_size=(1, 1, 1), data_format='channels_first')(concatenate([t1, t2], axis=1))
    full = PReLU()(full)
    full = Conv3D(dense_size/2, kernel_size=(1, 1, 1), data_format='channels_first')(full)
    full = PReLU()(full)
    full = Conv3D(dense_size/4, kernel_size=(1, 1, 1), data_format='channels_first')(full)
    full = PReLU()(full)
    full = Conv3D(4, kernel_size=(1, 1, 1), data_format='channels_first')(full)

    rf = full

    while np.product(K.int_shape(rf)[2:]) > 1:
        rf = Conv3D(4, kernel_size=(3, 3, 3), data_format='channels_first')(rf)
    rf = Flatten()(rf)

    # FC labeling
    full = Reshape((4, -1))(full)
    full = Permute((2, 1))(full)
    full_out = Activation('softmax', name='fc_out')(full)

    # Final labeling
    merged = concatenate([t2_f, t1_f, PReLU()(csf), PReLU()(gm), PReLU()(wm), PReLU()(rf)])
    merged = Dropout(0.5)(merged)
    brain = Dense(4, name='brain', activation='softmax')(merged)

    # Weights and outputs
    weights = [0.2,     0.5,    0.5,    1.0,   0.8]
    outputs = [csf_out, gm_out, wm_out, brain, full_out]

    return compile_network(merged_inputs, outputs, weights)


def get_brats_net(input_shape, filters_list, kernel_size_list, dense_size, nlabels, domain=False):
    inputs = Input(shape=input_shape, name='merged_inputs')
    conv = inputs
    for filters, kernel_size in zip(filters_list, kernel_size_list):
        conv = Conv3D(filters, kernel_size=kernel_size, activation='relu', data_format='channels_first')(conv)
        conv = Dropout(0.5)(conv)

    if not domain:
        full = Conv3D(dense_size, kernel_size=(1, 1, 1), data_format='channels_first', name='fc_dense')(conv)
        full = PReLU()(Dropout(0.5)(full))
        full = Conv3D(nlabels, kernel_size=(1, 1, 1), data_format='channels_first', name='fc')(full)

        rf = concatenate([conv, full], axis=1)

        rf_num = 1
        while np.product(K.int_shape(rf)[2:]) > 1:
            rf = Conv3D(dense_size, kernel_size=(3, 3, 3), data_format='channels_first', name='rf_%d' % rf_num)(rf)
            rf = Dropout(0.5)(rf)
            rf_num += 1

        full = Reshape((nlabels, -1))(full)
        full = Permute((2, 1))(full)
        full_out = Activation('softmax', name='fc_out')(full)

        tumor = Dense(nlabels, activation='softmax', name='tumor')(Flatten()(rf))

        outputs = [tumor, full_out]

        net = Model(inputs=inputs, outputs=outputs)

        net.compile(
            optimizer='adadelta',
            loss='categorical_crossentropy',
            loss_weights=[0.8, 1.0],
            metrics=['accuracy']
        )

    else:
        net = Model(inputs=inputs, outputs=conv)

        net.compile(optimizer='adadelta', loss='mean_squared_error', metrics=['accuracy'])

    return net


def get_brats_old_domain(patch_size, filters_list, kernel_size_list):
    # This architecture is based on the functional Keras API to introduce 3 output paths:
    # - Whole tumor segmentation
    # - Core segmentation (including whole tumor)
    # - Whole segmentation (tumor, core and enhancing parts)
    # The idea is to let the network work on the three parts to improve the multiclass segmentation.
    merged_inputs = Input(shape=(4,) + patch_size, name='merged_inputs')
    flair = Reshape((1,) + patch_size)(
        Lambda(
            lambda l: l[:, 0, :, :, :],
            output_shape=(1,) + patch_size)(merged_inputs),
    )
    t2 = Reshape((1,) + patch_size)(
        Lambda(lambda l: l[:, 1, :, :, :], output_shape=(1,) + patch_size)(merged_inputs)
    )
    t1 = Lambda(lambda l: l[:, 2:, :, :, :], output_shape=(2,) + patch_size)(merged_inputs)
    for filters, kernel_size in zip(filters_list[:-1], kernel_size_list[:-1]):
        flair = Conv3D(filters,
                       kernel_size=kernel_size,
                       activation='relu',
                       data_format='channels_first'
                       )(flair)
        t2 = Conv3D(filters,
                    kernel_size=kernel_size,
                    activation='relu',
                    data_format='channels_first'
                    )(t2)
        t1 = Conv3D(filters,
                    kernel_size=kernel_size,
                    activation='relu',
                    data_format='channels_first'
                    )(t1)
        flair = Dropout(0.5)(flair)
        t2 = Dropout(0.5)(t2)
        t1 = Dropout(0.5)(t1)

    flair = Conv3D(filters_list[-1],
                   kernel_size=kernel_size_list[-1],
                   activation='relu',
                   data_format='channels_first',
                   name='flair'
                   )(flair)
    t2 = Conv3D(filters_list[-1],
                kernel_size=kernel_size_list[-1],
                activation='relu',
                data_format='channels_first',
                name='t2'
                )(t2)
    t1 = Conv3D(filters_list[-1],
                kernel_size=kernel_size_list[-1],
                activation='relu',
                data_format='channels_first',
                name='t1'
                )(t1)

    net = Model(inputs=merged_inputs, outputs=[flair, t2, t1])

    net.compile(optimizer='adadelta', loss='mean_squared_error', metrics=['accuracy'])

    return net
