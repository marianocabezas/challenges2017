from __future__ import print_function
import argparse
import os
from time import strftime
import numpy as np
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Conv3D, Dropout, Flatten, Input, concatenate, Reshape, Lambda
from keras.layers import BatchNormalization, LSTM, Permute, Activation, PReLU
from nibabel import load as load_nii
from nibabel import save as save_nii
from utils import color_codes, nfold_cross_validation
from itertools import izip
from data_creation import load_patch_batch_train, get_cnn_centers
from data_creation import load_patch_batch_generator_test
from data_manipulation.generate_features import get_mask_voxels
from data_manipulation.metrics import dsc_seg


def parse_inputs():
    # I decided to separate this function, for easier acces to the command line parameters
    parser = argparse.ArgumentParser(description='Test different nets with 3D data.')
    parser.add_argument('-f', '--folder', dest='dir_name', default='/home/mariano/images/iSeg2017/Training/')
    parser.add_argument('-i', '--patch-width', dest='patch_width', type=int, default=17)
    parser.add_argument('-k', '--kernel-size', dest='conv_width', nargs='+', type=int, default=3)
    parser.add_argument('-c', '--conv-blocks', dest='conv_blocks', type=int, default=5)
    parser.add_argument('-b', '--batch-size', dest='batch_size', type=int, default=1024)
    parser.add_argument('-d', '--dense-size', dest='dense_size', type=int, default=256)
    parser.add_argument('-D', '--down-factor', dest='dfactor', type=int, default=1)
    parser.add_argument('-n', '--num-filters', action='store', dest='n_filters', nargs='+', type=int, default=[32])
    parser.add_argument('-e', '--epochs', action='store', dest='epochs', type=int, default=50)
    parser.add_argument('-E', '--experimental', action='store', dest='experimental', type=int, default=None)
    parser.add_argument('-q', '--queue', action='store', dest='queue', type=int, default=100)
    parser.add_argument('-s', '--sequential', action='store_true', dest='sequential', default=False)
    parser.add_argument('--preload', action='store_true', dest='preload', default=False)
    parser.add_argument('--t1', action='store', dest='t1', default='-T1.hdr')
    parser.add_argument('--t2', action='store', dest='t2', default='-T2.hdr')
    parser.add_argument('--labels', action='store', dest='labels', default='-label.hdr')
    return vars(parser.parse_args())


def list_directories(path):
    return filter(os.path.isdir, [os.path.join(path, f) for f in os.listdir(path)])


def get_names_from_path(options):
    path = options['dir_name']

    patients = range(10)

    # Prepare the names
    t1_names = [os.path.join(path, 'subject-%d' % (p+1) + options['t1']) for p in patients]
    t2_names = [os.path.join(path, 'subject-%d' % (p+1) + options['t2']) for p in patients]
    label_names = np.array([os.path.join(path, 'subject-%d' % (p+1) + options['labels']) for p in patients])
    image_names = np.stack([t1_names, t2_names], axis=1)

    return image_names, label_names


def get_convolutional_block(input_l, filters_list, kernel_size_list, activation=PReLU, drop=0.5):
    for filters, kernel_size in zip(filters_list, kernel_size_list):
        input_l = Conv3D(filters, kernel_size=kernel_size, data_format='channels_first')(input_l)
        input_l = BatchNormalization(axis=1)(input_l)
        input_l = activation()(input_l)
        input_l = Dropout(drop)(input_l)

    return input_l


def get_network_1(merged_inputs, patch_size, filters_list, kernel_size_list, dense_size):
    # Input splitting
    t1 = Reshape((1,) + patch_size)(
        Lambda(lambda l: l[:, 0, :, :, :], output_shape=(1,) + patch_size)(merged_inputs)
    )
    t2 = Reshape((1,) + patch_size)(
        Lambda(lambda l: l[:, 1, :, :, :], output_shape=(1,) + patch_size)(merged_inputs)
    )

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
    csf = Dense(2)(merged)
    gm = Dense(2)(merged)
    wm = Dense(2)(merged)
    csf_out = Activation('softmax', name='csf')(csf)
    gm_out = Activation('softmax', name='gm')(gm)
    wm_out = Activation('softmax', name='wm')(wm)

    # Final labeling
    merged = concatenate([t2_f, t1_f, PReLU()(csf), PReLU()(gm), PReLU()(wm)])
    merged = Dropout(0.5)(merged)
    brain = Dense(4, name='brain', activation='softmax')(merged)

    # Weights and outputs
    weights = [0.2, 0.5, 0.5, 1.0]
    outputs = [csf_out, gm_out, wm_out, brain]

    return weights, outputs


def get_network_2(merged_inputs, filters_list, kernel_size_list, dense_size):
    # Convolutional part
    merged = get_convolutional_block(merged_inputs, filters_list, kernel_size_list)

    # LSTM stuff
    patch_center = Reshape((filters_list[-1], -1))(merged)
    patch_center = Dense(4, name='pre_rf')(Permute((2, 1))(patch_center))
    rf = LSTM(4, implementation=1)(patch_center)
    rf_out = Activation('softmax', name='rf_out')(rf)
    rf = PReLU(name='rf')(rf)

    # Tissue binary stuff
    merged_f = Flatten()(merged)
    merged_f = Dense(dense_size, activation='relu')(merged_f)
    merged_f = Dropout(0.5)(merged_f)
    csf = Dense(2)(merged_f)
    gm = Dense(2)(merged_f)
    wm = Dense(2)(merged_f)
    csf_out = Activation('softmax', name='csf')(csf)
    gm_out = Activation('softmax', name='gm')(gm)
    wm_out = Activation('softmax', name='wm')(wm)

    # Brain labeling
    merged = concatenate([PReLU(csf), PReLU(gm), PReLU(wm), merged_f])
    merged = Dropout(0.5)(merged)
    brain = Dense(4)(merged)
    brain_out = Activation('softmax', name='brain_out')(brain)
    brain = PReLU(name='brain')(brain)

    # Final labeling
    final_layers = concatenate([
        Dropout(0.5)(brain),
        Dropout(0.5)(rf),
    ])
    final = Dense(4, name='merge', activation='softmax')(final_layers)

    # Weights and outputs
    weights = [0.2, 0.5, 0.5, 0.8, 0.8, 1.0]
    outputs = [csf_out, gm_out, wm_out, brain_out, rf_out, final]

    return weights, outputs


def get_network_3(merged_inputs, filters_list, kernel_size_list, dense_size):
    # Convolutional stuff
    merged = get_convolutional_block(merged_inputs, filters_list, kernel_size_list)

    # Tissue binary stuff
    merged_f = Flatten()(merged)
    merged_f = Dense(dense_size, activation='relu')(merged_f)
    merged_f = Dropout(0.5)(merged_f)
    csf = Dense(2)(merged_f)
    gm = Dense(2)(merged_f)
    wm = Dense(2)(merged_f)
    csf_out = Activation('softmax', name='csf')(csf)
    gm_out = Activation('softmax', name='gm')(gm)
    wm_out = Activation('softmax', name='wm')(wm)

    # Final labeling stuff
    merged = concatenate([PReLU(csf), PReLU(gm), PReLU(wm), merged_f])
    merged = Dropout(0.5)(merged)
    brain = Dense(4, activation='softmax', name='brain')(merged)

    # Weights and outputs
    weights = [0.2, 0.5, 0.5, 1.0]
    outputs = [csf_out, gm_out, wm_out, brain]

    return weights, outputs


def main():
    options = parse_inputs()
    c = color_codes()

    # Prepare the net architecture parameters
    dfactor = options['dfactor']
    # Prepare the net hyperparameters
    epochs = options['epochs']
    patch_width = options['patch_width']
    patch_size = (patch_width, patch_width, patch_width)
    batch_size = options['batch_size']
    dense_size = options['dense_size']
    conv_blocks = options['conv_blocks']
    n_filters = options['n_filters']
    filters_list = n_filters if len(n_filters) > 1 else n_filters*conv_blocks
    conv_width = options['conv_width']
    kernel_size_list = conv_width if isinstance(conv_width, list) else [conv_width]*conv_blocks
    experimental = options['experimental']
    # Data loading parameters
    preload = options['preload']
    queue = options['queue']

    # Prepare the sufix that will be added to the results for the net and images
    path = options['dir_name']
    filters_s = 'n'.join(['%d' % nf for nf in filters_list])
    conv_s = 'c'.join(['%d' % cs for cs in kernel_size_list])
    exp_s = 'experimental-%d' % experimental if experimental else 'baseline'
    params_s = (exp_s, dfactor, patch_width, conv_s, filters_s, dense_size, epochs)
    sufix = '%s.D%d.p%d.c%s.n%s.d%d.e%d.' % params_s

    exp_s = c['b'] + '(experimental)' if experimental else c['b'] + '(baseline)'
    print(c['c'] + '[' + strftime("%H:%M:%S") + '] ' + 'Starting cross-validation ' + exp_s + c['nc'])
    # N-fold cross validation main loop (we'll do 2 training iterations with testing for each patient)
    data_names, label_names = get_names_from_path(options)
    folds = len(data_names)
    fold_generator = izip(nfold_cross_validation(data_names, label_names, n=folds, val_data=0.25), xrange(folds))
    dsc_results = list()
    for (train_data, train_labels, val_data, val_labels, test_data, test_labels), i in fold_generator:
        print(c['c'] + '[' + strftime("%H:%M:%S") + ']  ' + c['nc'] + 'Fold %d/%d: ' % (i+1, folds) + c['g'] +
              'Number of training/validation/testing images (%d=%d/%d=%d/%d)'
              % (len(train_data), len(train_labels), len(val_data), len(val_labels), len(test_data)) + c['nc'])
        # Prepare the data relevant to the leave-one-out (subtract the patient from the dataset and set the path)
        # Also, prepare the network
        net_name = os.path.join(path, 'iseg2017.fold%d' % i + sufix + 'mdl')

        # First we check that we did not train for that patient, in order to save time
        try:
            net = keras.models.load_model(net_name)
        except IOError:
            # NET definition using Keras
            train_centers = get_cnn_centers(train_data[:, 0], train_labels)
            val_centers = get_cnn_centers(val_data[:, 0], val_labels)
            train_samples = len(train_centers)/dfactor
            val_samples = len(val_centers) / dfactor
            print(c['c'] + '[' + strftime("%H:%M:%S") + ']    ' + c['g'] + 'Creating and compiling the model ' +
                  c['b'] + '(%d samples)' % train_samples + c['nc'])
            train_steps_per_epoch = -(-train_samples/batch_size)
            val_steps_per_epoch = -(-val_samples / batch_size)
            input_shape = (2,) + patch_size
            # This architecture is based on the functional Keras API to introduce 3 output paths:
            # - Whole tumor segmentation
            # - Core segmentation (including whole tumor)
            # - Whole segmentation (tumor, core and enhancing parts)
            # The idea is to let the network work on the three parts to improve the multiclass segmentation.
            merged_inputs = Input(shape=input_shape, name='merged_inputs')

            if experimental:
                network_func = [get_network_2, get_network_3]
                outputs, weights = network_func[experimental-1](
                    merged_inputs,
                    filters_list,
                    kernel_size_list,
                    dense_size
                )
            else:
                outputs, weights = get_network_1(
                    merged_inputs,
                    patch_size,
                    filters_list,
                    kernel_size_list,
                    dense_size
                )

            net = Model(inputs=merged_inputs, outputs=outputs)

            net.compile(
                optimizer='adadelta',
                loss='categorical_crossentropy',
                loss_weights=weights,
                metrics=['accuracy']
            )

            print(c['c'] + '[' + strftime("%H:%M:%S") + ']    ' +
                  c['g'] + 'Training the model with a generator for ' +
                  c['b'] + '(%d parameters)' % net.count_params() + c['nc'])
            print(net.summary())
            net.fit_generator(
                generator=load_patch_batch_train(
                    image_names=train_data,
                    label_names=train_labels,
                    centers=train_centers,
                    batch_size=batch_size,
                    size=patch_size,
                    nlabels=4,
                    dfactor=dfactor,
                    preload=preload,
                    split=True,
                    iseg=True,
                    experimental=experimental,
                    datatype=np.float32
                ),
                validation_data=load_patch_batch_train(
                    image_names=val_data,
                    label_names=val_labels,
                    centers=val_centers,
                    batch_size=batch_size,
                    size=patch_size,
                    nlabels=4,
                    dfactor=dfactor,
                    preload=preload,
                    split=True,
                    iseg=True,
                    experimental=experimental,
                    datatype=np.float32
                ),
                steps_per_epoch=train_steps_per_epoch,
                validation_steps=val_steps_per_epoch,
                max_q_size=queue,
                epochs=epochs
            )
            net.save(net_name)

        # Then we test the net.
        for p, gt_name in zip(test_data, test_labels):
            p_name = '-'.join(p[0].rsplit('/')[-1].rsplit('.')[0].rsplit('-')[:-1])
            patient_path = '/'.join(p[0].rsplit('/')[:-1])
            outputname = os.path.join(patient_path, method_name + p_name + 'brain.roi.hdr')
            gt_nii = load_nii(gt_name)
            gt = np.copy(np.squeeze(gt_nii.get_data()))
            vals = np.unique(gt.flatten())
            try:
                image = np.squeeze(load_nii(outputname).get_data())
            except IOError:
                roi = np.squeeze(load_nii(p[0]).get_data())
                centers = get_mask_voxels(roi.astype(dtype=np.bool))
                test_samples = np.count_nonzero(roi)
                image = np.zeros_like(roi).astype(dtype=np.uint8)
                print(c['c'] + '[' + strftime("%H:%M:%S") + ']    ' + c['g'] +
                      '<Creating the probability map ' + c['b'] + p_name + c['nc'] + c['g'] +
                      ' (%d samples)>' % test_samples + c['nc'])
                test_steps_per_epoch = -(-test_samples / batch_size)
                y_pr_pred = net.predict_generator(
                    generator=load_patch_batch_generator_test(
                        image_names=p,
                        centers=centers,
                        batch_size=batch_size,
                        size=patch_size,
                        preload=preload,
                    ),
                    steps=test_steps_per_epoch,
                    max_q_size=queue
                )
                [x, y, z] = np.stack(centers, axis=1)

                for num, results in enumerate(y_pr_pred):
                    brain = np.argmax(results, axis=1)
                    image[x, y, z] = brain
                    if num is 0:
                        im = sufix + 'csf.'
                        gt_nii.get_data()[:] = np.expand_dims(image, axis=3)
                    elif num is 1:
                        im = sufix + 'gm.'
                        gt_nii.get_data()[:] = np.expand_dims(image, axis=3)
                    elif num is 2:
                        im = sufix + 'wm.'
                        gt_nii.get_data()[:] = np.expand_dims(image, axis=3)
                    elif num is 3:
                        im = sufix + 'brain.'
                        gt_nii.get_data()[:] = np.expand_dims(vals[image], axis=3)
                    elif num is 4:
                        im = sufix + 'rf.'
                        gt_nii.get_data()[:] = np.expand_dims(vals[image], axis=3)
                    else:
                        im = sufix + 'merge.'
                        gt_nii.get_data()[:] = np.expand_dims(vals[image], axis=3)
                    roiname = os.path.join(patient_path, 'deep-' + p_name + im + 'roi.img')
                    print(c['g'] + '                   -- Saving image ' + c['b'] + roiname + c['nc'])
                    save_nii(gt_nii, roiname)

                y_pred = np.argmax(y_pr_pred[-1], axis=1)

                image[x, y, z] = y_pred

            gt_mask = np.sum(
                map(lambda (l, val): np.array(gt == val, dtype=np.uint8) * l, enumerate(vals)), axis=0
            )
            results = (
                p_name,
                dsc_seg(gt_mask == 1, image == 1),
                dsc_seg(gt_mask == 2, image == 2),
                dsc_seg(gt_mask == 3, image == 3)
            )
            dsc_results.append(results)
            print('Subject %s DSC: %f/%f/%f' % results)

    f_dsc = tuple(np.array(dsc_results).mean())
    print('Final results DSC: %f/%f/%f' % f_dsc)


if __name__ == '__main__':
    main()
