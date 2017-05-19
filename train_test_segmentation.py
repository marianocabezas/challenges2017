from __future__ import print_function
import argparse
import os
from time import strftime
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv3D, Dropout
from nibabel import load as load_nii
from utils import color_codes, nfold_cross_validation
from itertools import izip
from data_creation import load_patch_batch, get_cnn_rois


def parse_inputs():
    # I decided to separate this function, for easier acces to the command line parameters
    parser = argparse.ArgumentParser(description='Test different nets with 3D data.')
    parser.add_argument('-f', '--folder', dest='dir_name', default='/home/mariano/DATA/Brats17TrainingData/')
    parser.add_argument('-F', '--n-fold', dest='folds', type=int, default=10)
    parser.add_argument('-i', '--patch-width', dest='patch_width', type=int, default=9)
    parser.add_argument('-p', '--pool-size', dest='pool_size', type=int, default=2)
    parser.add_argument('-k', '--kernel-size', dest='conv_width', nargs='+', type=int, default=3)
    parser.add_argument('-c', '--conv-blocks', dest='conv_blocks', type=int, default=2)
    parser.add_argument('-b', '--batch-size', dest='batch_size', type=int, default=10000)
    parser.add_argument('-d', '--dense-size', dest='dense_size', type=int, default=256)
    parser.add_argument('-n', '--num-filters', action='store', dest='n_filters', nargs='+', type=int, default=[32])
    parser.add_argument('-e', '--epochs', action='store', dest='epochs', type=int, default=1000)
    parser.add_argument('--padding', action='store', dest='padding', default='valid')
    parser.add_argument('--no-flair', action='store_false', dest='use_flair', default=True)
    parser.add_argument('--no-t1', action='store_false', dest='use_t1', default=True)
    parser.add_argument('--no-t1ce', action='store_false', dest='use_t1ce', default=True)
    parser.add_argument('--no-t2', action='store_false', dest='use_t2', default=True)
    parser.add_argument('--flair', action='store', dest='flair', default='_flair.nii.gz')
    parser.add_argument('--t1', action='store', dest='t1', default='_t1.nii.gz')
    parser.add_argument('--t1ce', action='store', dest='t1ce', default='_t1ce.nii.gz')
    parser.add_argument('--t2', action='store', dest='t2', default='_t2.nii.gz')
    parser.add_argument('--labels', action='store', dest='labels', default='_seg.nii.gz')
    parser.add_argument('-m', '--multi-channel', action='store_true', dest='multi', default=False)
    return vars(parser.parse_args())


def list_directories(path):
    return filter(os.path.isdir, [os.path.join(path, f) for f in os.listdir(path)])


def get_names_from_path(options):
    path = options['dir_name']

    patients = sorted(np.concatenate([list_directories(f) for f in list_directories(path)]))

    # Prepare the names
    flair_names = [os.path.join(path, p, p.split('/')[-1] + options['flair'])
                   for p in patients] if options['use_flair'] else None
    t1_names = [os.path.join(path, p, p.split('/')[-1] + options['t1'])
                for p in patients] if options['use_t1'] else None
    t1ce_names = [os.path.join(path, p, p.split('/')[-1] + options['t1ce'])
                  for p in patients] if options['use_t1ce'] else None
    t2_names = [os.path.join(path, p, p.split('/')[-1] + options['t2'])
                for p in patients] if options['use_t2'] else None
    label_names = np.array([os.path.join(path, p, p.split('/')[-1] + options['labels']) for p in patients])
    image_names = np.stack(filter(None, [flair_names, t1_names, t1ce_names, t2_names]), axis=1)

    return image_names, label_names


def main():
    options = parse_inputs()
    c = color_codes()

    # Prepare the net architecture parameters
    multi = options['multi']
    # Prepare the net hyperparameters
    epochs = options['epochs']
    padding = options['padding']
    patch_width = options['patch_width']
    patch_size = (patch_width, patch_width, patch_width)
    pool_size = options['pool_size']
    batch_size = options['batch_size']
    dense_size = options['dense_size']
    conv_blocks = options['conv_blocks']
    n_filters = options['n_filters']
    n_filters = n_filters if len(n_filters) > 1 else n_filters*conv_blocks
    conv_width = options['conv_width']
    conv_size = conv_width if isinstance(conv_width, list) else [conv_width]*conv_blocks

    # Prepare the sufix that will be added to the results for the net and images
    path = options['dir_name']
    filters_s = 'n'.join(['%d' % nf for nf in n_filters])
    conv_s = 'c'.join(['%d' % cs for cs in conv_size])
    mc_s = '.mc' if multi else ''
    sufix = '%s.p%d.c%s.n%s.d%d.e%d.pad_%s' % (mc_s, patch_width, conv_s, filters_s, dense_size, epochs, padding)
    n_channels = np.count_nonzero([
        options['use_flair'],
        options['use_t1'],
        options['use_t1ce'],
        options['use_t2']]
    )

    print(c['c'] + '[' + strftime("%H:%M:%S") + '] ' + 'Starting leave-one-out' + c['nc'])
    # N-fold cross validation main loop (we'll do 2 training iterations with testing for each patient)
    data_names, label_names = get_names_from_path(options)
    folds = options['folds']
    fold_generator = izip(nfold_cross_validation(data_names, label_names), xrange(folds))
    for (training_data, training_labels, testing_data), i in fold_generator:
        # Prepare the data relevant to the leave-one-out (subtract the patient from the dataset and set the path)
        # Also, prepare the network
        print(c['c'] + '[' + strftime("%H:%M:%S") + ']  ' + c['nc'] + 'Fold ' + c['g'] +
              '%d/%d' % (i+1, folds) + c['nc'])
        net_name = os.path.join(path, 'baseline-brats2017.fold%d' % i + sufix + '.')

        # First we check that we did not train for that patient, in order to save time
        try:
            net = keras.models.load_model(net_name)
        except IOError:
            print(c['c'] + '[' + strftime("%H:%M:%S") + ']    ' +
                  c['g'] + 'Creating and compiling the model for ' + c['b'] + 'iteration 1' + c['nc'])
            # NET definition using Keras
            rois = get_cnn_rois(training_data[:, 0], training_labels)
            steps_per_epoch = -(-sum([np.count_nonzero(roi) * 2 for roi in rois])/batch_size)
            input_shape = (n_channels,) + patch_size
            kernel_size = (conv_width,) * 3
            filters = options['n_filters'][0]
            net = Sequential()
            net.add(Conv3D(
                filters,
                kernel_size=kernel_size,
                input_shape=input_shape,
                activation='relu',
                data_format='channels_first'
            ))
            net.add(Dropout(0.5))
            net.add(Conv3D(filters, kernel_size=kernel_size, activation='relu'))
            net.add(Dropout(0.5))
            net.add(Conv3D(filters, kernel_size=kernel_size, activation='relu'))
            net.add(Dropout(0.5))
            net.add(Dense(dense_size, activation='relu'))
            net.add(Dropout(0.5))
            net.add(Dense(5, activation='softmax'))
            net.compile(optimizer='adadelta',
                        loss='categorical_crossentropy',
                        metrics=['accuracy']
                        )

            print(c['c'] + '[' + strftime("%H:%M:%S") + ']    ' +
                  c['g'] + 'Training th emodel with a generator for ' + c['b'] + 'iteration 1' + c['nc'])
            net.fit_generator(
                generator=load_patch_batch(
                    training_data,
                    training_labels,
                    rois,
                    batch_size,
                    patch_size,
                    datatype=np.float32
                ),
                steps_per_epoch=steps_per_epoch,
                epochs=epochs
            )
            net.save(net_name)

        # Then we test the net. Again we save time by checking if we already tested that patient.
        for p in testing_data:
            patient_path = '/'.join(p[0].rsplit('/')[:-1])
            outputname = os.path.join(patient_path, sufix + 'test.nii.gz')
            try:
                load_nii(outputname)
            except IOError:
                print(c['c'] + '[' + strftime("%H:%M:%S") + ']    ' + c['g'] +
                      '<Creating the probability map ' + c['b'] + '1' + c['nc'] + c['g'] + '>' + c['nc'])
                # TODO: Net testing


if __name__ == '__main__':
    main()
