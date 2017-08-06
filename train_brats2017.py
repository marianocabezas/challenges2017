from __future__ import print_function
import argparse
import os
from time import strftime
import numpy as np
from keras.models import Model
from keras.layers import Dense, Conv3D, Dropout, Flatten, PReLU, Input, Reshape, Permute, Activation, concatenate
from utils import color_codes
from data_creation import load_patch_batch_train, get_cnn_centers, load_patches_train


def parse_inputs():
    # I decided to separate this function, for easier acces to the command line parameters
    parser = argparse.ArgumentParser(description='Test different nets with 3D data.')
    parser.add_argument('-f', '--folder', dest='dir_name', default='/home/mariano/DATA/Brats17Test-Training/')
    parser.add_argument('-F', '--n-fold', dest='folds', type=int, default=5)
    parser.add_argument('-i', '--patch-width', dest='patch_width', type=int, default=17)
    parser.add_argument('-k', '--kernel-size', dest='conv_width', nargs='+', type=int, default=3)
    parser.add_argument('-c', '--conv-blocks', dest='conv_blocks', type=int, default=4)
    parser.add_argument('-b', '--batch-size', dest='batch_size', type=int, default=1024)
    parser.add_argument('-d', '--dense-size', dest='dense_size', type=int, default=256)
    parser.add_argument('-D', '--down-factor', dest='dfactor', type=int, default=50)
    parser.add_argument('-n', '--num-filters', action='store', dest='n_filters', nargs='+', type=int, default=[32])
    parser.add_argument('-e', '--epochs', action='store', dest='epochs', type=int, default=50)
    parser.add_argument('-q', '--queue', action='store', dest='queue', type=int, default=10)
    parser.add_argument('-v', '--validation-rate', action='store', dest='val_rate', type=float, default=0.25)
    parser.add_argument('-u', '--unbalanced', action='store_false', dest='balanced', default=True)
    parser.add_argument('-s', '--sequential', action='store_true', dest='sequential', default=False)
    parser.add_argument('-r', '--recurrent', action='store_true', dest='recurrent', default=False)
    parser.add_argument('-p', '--preload', action='store_true', dest='preload', default=False)
    parser.add_argument('--flair', action='store', dest='flair', default='_flair.nii.gz')
    parser.add_argument('--t1', action='store', dest='t1', default='_t1.nii.gz')
    parser.add_argument('--t1ce', action='store', dest='t1ce', default='_t1ce.nii.gz')
    parser.add_argument('--t2', action='store', dest='t2', default='_t2.nii.gz')
    parser.add_argument('--labels', action='store', dest='labels', default='_seg.nii.gz')
    return vars(parser.parse_args())


def list_directories(path):
    return filter(os.path.isdir, [os.path.join(path, f) for f in os.listdir(path)])


def get_names_from_path(options):
    path = options['dir_name']

    patients = sorted(list_directories(path))

    # Prepare the names
    flair_names = [os.path.join(path, p, p.split('/')[-1] + options['flair']) for p in patients]
    t2_names = [os.path.join(path, p, p.split('/')[-1] + options['t2']) for p in patients]
    t1_names = [os.path.join(path, p, p.split('/')[-1] + options['t1']) for p in patients]
    t1ce_names = [os.path.join(path, p, p.split('/')[-1] + options['t1ce']) for p in patients]

    label_names = np.array([os.path.join(path, p, p.split('/')[-1] + options['labels']) for p in patients])
    image_names = np.stack(filter(None, [flair_names, t2_names, t1_names, t1ce_names]), axis=1)

    return image_names, label_names


def main():
    options = parse_inputs()
    c = color_codes()

    # Prepare the net architecture parameters
    dfactor = options['dfactor']
    # Prepare the net hyperparameters
    num_classes = 4
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
    balanced = options['balanced']
    # Data loading parameters
    preload = options['preload']
    queue = options['queue']

    # Prepare the sufix that will be added to the results for the net and images
    path = options['dir_name']
    filters_s = 'n'.join(['%d' % nf for nf in filters_list])
    conv_s = 'c'.join(['%d' % cs for cs in kernel_size_list])
    ub_s = '.ub' if not balanced else ''
    params_s = (ub_s, dfactor, patch_width, conv_s, filters_s, dense_size, epochs)
    sufix = '%s.D%d.p%d.c%s.n%s.d%d.e%d.' % params_s
    n_channels = 4
    preload_s = ' (with ' + c['b'] + 'preloading' + c['nc'] + c['c'] + ')' if preload else ''

    print(c['c'] + '[' + strftime("%H:%M:%S") + '] ' + 'Starting training' + preload_s + c['nc'])
    # N-fold cross validation main loop (we'll do 2 training iterations with testing for each patient)
    data_names, label_names = get_names_from_path(options)

    shuffled_indices = np.random.permutation(xrange(len(data_names)))
    data_names = data_names[shuffled_indices]
    label_names = label_names[shuffled_indices]

    val_rate = options['val_rate']

    val_len = int(len(data_names) * val_rate)
    train_data = data_names[val_len:]
    train_labels = label_names[val_len:]
    val_data = data_names[:val_len]
    val_labels = label_names[:val_len]

    print(c['c'] + '[' + strftime("%H:%M:%S") + ']  ' + c['nc'] + c['g'] +
          'Number of training/validation images (%d=%d/%d=%d)' %
          (len(train_data), len(train_labels), len(val_data), len(val_labels)) + c['nc'])
    #  Also, prepare the network
    net_name = os.path.join(path, 'CBICA-brats2017' + sufix + 'mdl')

    train_centers = get_cnn_centers(train_data[:, 0], train_labels, balanced=balanced)
    val_centers = get_cnn_centers(val_data[:, 0], val_labels, balanced=balanced)
    train_samples = len(train_centers)/dfactor
    val_samples = len(val_centers) / dfactor
    print(c['c'] + '[' + strftime("%H:%M:%S") + ']    ' + c['g'] + 'Creating and compiling the model ' +
          c['b'] + '(%d samples)' % train_samples + c['nc'])
    train_steps_per_epoch = -(-train_samples/batch_size)
    val_steps_per_epoch = -(-val_samples / batch_size)
    input_shape = (data_names.shape[1],) + patch_size

    # Sequential model that merges all 4 images. This architecture is just a set of convolutional blocks
    #  that end in a dense layer. This is supposed to be an original baseline.
    inputs = Input(shape=input_shape, name='merged_inputs')
    conv = inputs
    for filters, kernel_size in zip(filters_list, kernel_size_list):
        conv = Conv3D(filters, kernel_size=kernel_size, activation='relu', data_format='channels_first')(conv)
        conv = Dropout(0.5)(conv)

    full = Conv3D(dense_size, kernel_size=(1, 1, 1), data_format='channels_first')(conv)
    full = PReLU()(full)
    full = Conv3D(2, kernel_size=(1, 1, 1), data_format='channels_first')(full)
    full = Reshape((2, -1))(full)
    full = Permute((2, 1))(full)
    full_out = Activation('softmax', name='fc_out')(full)

    dense = concatenate([Flatten()(conv), Flatten()(full)])
    dense = Dense(dense_size, activation='relu')(dense)
    dense = Dropout(0.5)(dense)
    tumor = Dense(2, activation='softmax')(dense)

    outputs = [tumor, full_out]

    net = Model(inputs=inputs, outputs=outputs)

    net.compile(
        optimizer='adadelta',
        loss='categorical_crossentropy',
        loss_weights=[0.8, 1.0],
        metrics=['accuracy']
    )

    print(c['c'] + '[' + strftime("%H:%M:%S") + ']    ' + c['g'] + 'Loading data ' +
          c['b'] + '(%d centers)' % len(train_centers) + c['nc'])
    fc_width = patch_width - sum(kernel_size_list) + conv_blocks
    fc_shape = (fc_width,) * 3
    x, y = load_patches_train(
        image_names=train_data,
        label_names=train_labels,
        centers=train_centers,
        size=patch_size,
        fc_shape=fc_shape,
        nlabels=2,
        dfactor=dfactor,
        preload=preload,
        split=True,
        iseg=True,
        experimental=1,
        datatype=np.float32
    )

    print(c['c'] + '[' + strftime("%H:%M:%S") + ']    ' +
          c['g'] + 'Training the model with a generator for ' +
          c['b'] + '(%d parameters)' % net.count_params() + c['nc'])
    print(net.summary())

    net.fit(x, y, batch_size=batch_size, validation_split=0.25, epochs=epochs)
    net.save(net_name)


if __name__ == '__main__':
    main()
