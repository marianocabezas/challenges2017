from keras import backend as K
from keras.layers import Dense, Conv3D, Dropout, Flatten, Input, concatenate, Reshape, Lambda
from keras.layers import BatchNormalization, LSTM, Permute, Activation, PReLU, Average, Cropping3D
from keras.models import Model
from itertools import product
import numpy as np
from layers import GradientReversal, PrimaryCap3D, CapsuleLayer, Length


def dsc_loss(y_true, y_pred):
    dsc_class = K.sum(y_true * y_pred, axis=0) / (K.sum(y_true, axis=0) + K.sum(y_pred, axis=0))
    return 1 - 2 * dsc_class[0]


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
    x_input = [lambda_x(PReLU()(full)) for (i, j) in x_combos] + [lambda_x_rev(PReLU()(full)) for (i, j) in x_combos]
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


def get_brats_gan_fc(input_shape, filters_list, kernel_size_list, dense_size, nlabels, lambda_var=None, capsule=False):
    s_inputs = Input(shape=input_shape, name='seg_inputs')
    d_inputs = Input(shape=input_shape, name='disc_inputs')

    inputs = [s_inputs, d_inputs]

    conv_s = s_inputs
    conv_d = d_inputs
    list_disc = []
    for i, (filters, kernel_size) in enumerate(zip(filters_list, kernel_size_list)):
        conv = Conv3D(filters, kernel_size=kernel_size, activation='relu', data_format='channels_first')
        conv_s = BatchNormalization(axis=1)(conv(conv_s))
        conv_d = BatchNormalization(axis=1)(conv(conv_d))
        list_disc.append(Cropping3D(cropping=len(filters_list) - i - 1, data_format='channels_first')(conv_d))

    full = Conv3D(dense_size, kernel_size=(1, 1, 1), data_format='channels_first', name='fc_dense')
    full_s = PReLU()(BatchNormalization(axis=1)(full(conv_s)))
    full = Conv3D(nlabels, kernel_size=(1, 1, 1), data_format='channels_first', name='fc')
    full_s = BatchNormalization(axis=1)(full(full_s))

    disc_input = concatenate(list_disc, axis=1)
    grad_reverse = GradientReversal(1) if lambda_var is None else GradientReversal(lambda_var)
    conv_d = Conv3D(
        filters_list[-1],
        kernel_size=kernel_size_list[-1],
        activation='relu',
        data_format='channels_first'
    )(grad_reverse(disc_input))
    conv_d = Conv3D(
        filters_list[-1],
        kernel_size=kernel_size_list[-1],
        activation='relu',
        data_format='channels_first'
    )(conv_d)
    conv_d = Conv3D(
        filters_list[-1],
        kernel_size=kernel_size_list[-1],
        activation='relu',
        data_format='channels_first'
    )(conv_d)
    conv_d = Conv3D(
        dense_size,
        kernel_size=(1, 1, 1),
        activation='relu',
        data_format='channels_first'
    )(conv_d)

    seg = Activation('softmax', name='seg')(
        Flatten()(
            Cropping3D(cropping=K.int_shape(full_s)[2]/2, data_format='channels_first')(full_s)
        )
    )
    disc = Dense(2, activation='softmax', name='disc')(Flatten()(conv_d))

    outputs = [seg, disc]

    gan_net = Model(inputs=inputs, outputs=outputs)
    seg_net = Model(inputs=s_inputs, outputs=seg)

    gan_net.compile(
        optimizer='adadelta',
        loss={'seg': 'categorical_crossentropy', 'disc': 'binary_crossentropy'},
        loss_weights=[1, 1],
        metrics=['accuracy']
    )

    return gan_net, seg_net


def get_brats_fc(input_shape, filters_list, kernel_size_list, dense_size, nlabels, capsule=False):
    s_inputs = Input(shape=input_shape, name='seg_inputs')

    conv_s = s_inputs
    for i, (filters, kernel_size) in enumerate(zip(filters_list, kernel_size_list)):
        conv = Conv3D(filters, kernel_size=kernel_size, activation='relu', data_format='channels_first')
        conv_s = BatchNormalization(axis=1)(conv(conv_s))

    full = Conv3D(dense_size, kernel_size=(1, 1, 1), data_format='channels_first', name='fc_dense')
    full_s = PReLU()(BatchNormalization(axis=1)(full(conv_s)))
    full = Conv3D(nlabels, kernel_size=(1, 1, 1), data_format='channels_first', name='fc')
    full_s = BatchNormalization(axis=1)(full(full_s))

    seg = Activation('softmax', name='seg')(
        Flatten()(
            Cropping3D(cropping=K.int_shape(full_s)[2]/2, data_format='channels_first')(full_s)
        )
    )

    seg_net = Model(inputs=s_inputs, outputs=seg)

    seg_net.compile(
        optimizer='adadelta',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return seg_net


def get_brats_caps(input_shape, filters_list, kernel_size_list, caps_size, nlabels):
    s_inputs = Input(shape=input_shape, name='seg_inputs')

    conv_s = s_inputs
    for i, (filters, kernel_size) in enumerate(zip(filters_list, kernel_size_list)):
        conv = Conv3D(filters, kernel_size=kernel_size, activation='relu', data_format='channels_first')
        conv_s = BatchNormalization(axis=1)(conv(conv_s))

    primarycaps = PrimaryCap3D(
        dim_vector=caps_size,
        filters=32,
        kernel_size=1,
        strides=2,
        padding='valid',
        name='primarycaps'
    )(conv_s)

    digitcaps = CapsuleLayer(
        num_capsule=nlabels,
        dim_vector=16,
        num_routing=3,
        name='digitcaps'
    )(primarycaps)

    seg = Length(name='capsnet')(digitcaps)

    seg_net = Model(inputs=s_inputs, outputs=seg)

    seg_net.compile(
        optimizer='adadelta',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return seg_net


def get_wmh_nets(input_shape, filters_list, kernel_size_list, dense_size, lambda_var, dsc_obj=False):
    s_inputs = Input(shape=input_shape, name='seg_inputs')
    d_inputs = Input(shape=input_shape, name='disc_inputs')
    obj_f = dsc_loss if dsc_obj else 'categorical_crossentropy'

    def convolutional_blocks(s, f_list, k_size_list, d=None, conv_list=None):
        for l, (filters, kernel_size) in enumerate(zip(f_list, k_size_list)):
            conv = Conv3D(filters, kernel_size=kernel_size, activation='relu', data_format='channels_first')
            s = BatchNormalization(axis=1)(conv(s))
            if d is not None:
                d = BatchNormalization(axis=1)(conv(d))
                d_crop = Cropping3D(cropping=len(filters_list) - l - 1, data_format='channels_first')(d)
                if conv_list is not None:
                    conv_list.append(d_crop)
        return s

    # < GAN creation >
    list_disc = []
    conv_s = convolutional_blocks(
        s=s_inputs,
        d=d_inputs,
        f_list=filters_list,
        k_size_list=kernel_size_list,
        conv_list=list_disc
    )

    dense_gan = Dense(dense_size, activation='relu')(Flatten()(conv_s))
    seg_gan = Dense(2, activation='softmax', name='seg')(dense_gan)
    # This network is only used for testing to skip the adversarial part. We don't need to compile because
    # we are already initialising the weights and layers in the main GAN.
    gan_test = Model(inputs=s_inputs, outputs=seg_gan)

    # Discriminator part
    disc_input = concatenate(list_disc, axis=1)
    grad_reverse = GradientReversal(lambda_var)
    conv_d = Conv3D(
        filters_list[-1],
        kernel_size=kernel_size_list[-1],
        activation='relu',
        data_format='channels_first'
    )(grad_reverse(disc_input))
    conv_d = Conv3D(
        filters_list[-1],
        kernel_size=kernel_size_list[-1],
        activation='relu',
        data_format='channels_first'
    )(conv_d)

    disc = Dense(2, activation='softmax', name='disc')(Flatten()(conv_d))

    gan = Model(inputs=[s_inputs, d_inputs], outputs=[seg_gan, disc])

    gan.compile(
        optimizer='adadelta',
        loss={'seg': dsc_loss, 'disc': 'binary_crossentropy'},
        loss_weights=[1, 1],
        metrics=['accuracy']
    )

    # < CNN creation >
    conv_cnn = convolutional_blocks(
        s=s_inputs,
        f_list=filters_list,
        k_size_list=kernel_size_list,
    )

    dense_cnn = Dense(dense_size, activation='relu')(Flatten()(conv_cnn))
    seg_cnn = Dense(2, activation='softmax', name='seg')(dense_cnn)

    cnn = Model(inputs=s_inputs, outputs=seg_cnn)

    cnn.compile(
        optimizer='adadelta',
        loss=dsc_loss,
        metrics=['accuracy']
    )

    return cnn, gan, gan_test
