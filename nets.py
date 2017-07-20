from keras import backend as K
from keras.layers import Dense, Conv3D, Dropout, Flatten, Input, concatenate, Reshape, Lambda
from keras.layers import BatchNormalization, LSTM, Permute, Activation, PReLU, Average
from keras.models import Model
from itertools import product


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
        input_l = BatchNormalization(axis=1)(input_l)
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

    full = Conv3D(dense_size, kernel_size=(1, 1, 1), data_format='channels_first')(concatenate([t1, t2]))
    full = PReLU()(full)
    full = Conv3D(dense_size/2, kernel_size=(1, 1, 1), data_format='channels_first')(full)
    full = PReLU()(full)
    full = Conv3D(dense_size/4, kernel_size=(1, 1, 1), data_format='channels_first')(full)
    full = PReLU()(full)
    full = Conv3D(4, kernel_size=(1, 1, 1), data_format='channels_first')(full)

    full_shape = K.int_shape(full)

    # # x LSTM
    # x_combos = product(range(full_shape[-2]), range(full_shape[-1]))
    # x_shape = full_shape[-2] * full_shape[-1]
    # lambda_x = Lambda(lambda l: K.reshape(l[:, :, :, i, j], (-1, 4, x_shape)), output_shape=(4, x_shape))
    # lambda_x_rev = Lambda(lambda l: K.reshape(l[:, :, -1::-1, i, j], (-1, 4, x_shape)), output_shape=(4, x_shape))
    # x_input = [lambda_x(PReLU()(full)) for (i, j) in x_combos] +[lambda_x_rev(PReLU()(full)) for (i, j) in x_combos]
    # x_lstm = [LSTM(4, implementation=1)(x) for x in x_input]
    #
    # # y LSTM
    # y_combos = product(range(full_shape[-3]), range(full_shape[-1]))
    # y_shape = full_shape[-3] * full_shape[-1]
    # lambda_y = Lambda(lambda l: K.reshape(l[:, :, i, :, j], (-1, 4, y_shape)), output_shape=(4, y_shape))
    # lambda_y_rev = Lambda(lambda l: K.reshape(l[:, :, i, -1::-1, j], (-1, 4, y_shape)), output_shape=(4, y_shape))
    # y_input = [lambda_y(PReLU()(full)) for (i, j) in y_combos] + [lambda_y_rev(PReLU()(full)) for (i, j) in y_combos]
    # y_lstm = [LSTM(4, implementation=1)(y) for y in y_input]
    #
    # # z LSTM
    # z_combos = product(range(full_shape[-3]), range(full_shape[-2]))
    # z_shape = full_shape[-3] * full_shape[-2]
    # lambda_z = Lambda(lambda l: K.reshape(l[:, :, i, j, :], (-1, 4, z_shape)), output_shape=(4, z_shape))
    # lambda_z_rev = Lambda(lambda l: K.reshape(l[:, :, i, j, -1::-1], (-1, 4, z_shape)), output_shape=(4, z_shape))
    # z_input = [lambda_z(PReLU()(full)) for (i, j) in z_combos] + [lambda_z_rev(PReLU()(full)) for (i, j) in z_combos]
    # z_lstm = [LSTM(4, implementation=1)(PReLU()(z)) for z in z_input]
    #
    # # Final LSTM
    # rf = Average()(x_lstm + y_lstm + z_lstm)
    #
    # FC labeling
    full = Reshape((4, -1))(full)
    full = Permute((2, 1))(full)
    full_out = Activation('softmax', name='fc_out')(full)
    # # rf = LSTM(4, implementation=1)(Reshape((4, -1))(full))
    rf = Dense(4)(Reshape((4, -1))(full))

    # Final labeling
    merged = concatenate([t2_f, t1_f, PReLU()(csf), PReLU()(gm), PReLU()(wm), PReLU()(full_out)])
    # merged = concatenate([t2_f, t1_f, PReLU()(csf), PReLU()(gm), PReLU()(wm), PReLU()(Flatten()(full))])
    merged = Dropout(0.5)(merged)
    brain = Dense(4, name='brain', activation='softmax')(merged)

    # Weights and outputs
    weights = [0.2,     0.5,    0.5,    1.0,   0.8]
    outputs = [csf_out, gm_out, wm_out, brain, full_out]

    return compile_network(merged_inputs, outputs, weights)
