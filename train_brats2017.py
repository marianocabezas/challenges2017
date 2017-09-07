from __future__ import print_function
import argparse
import os
from time import strftime
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from nets import get_brats_net
from utils import color_codes
from data_creation import get_cnn_centers, load_patches_train


def parse_inputs():
    # I decided to separate this function, for easier acces to the command line parameters
    parser = argparse.ArgumentParser(description='Test different nets with 3D data.')
    parser.add_argument('-f', '--folder', dest='dir_name', default='/home/mariano/DATA/Brats17Test-Training/')
    parser.add_argument('-F', '--n-fold', dest='folds', type=int, default=5)
    parser.add_argument('-i', '--patch-width', dest='patch_width', type=int, default=17)
    parser.add_argument('-k', '--kernel-size', dest='conv_width', nargs='+', type=int, default=3)
    parser.add_argument('-c', '--conv-blocks', dest='conv_blocks', type=int, default=4)
    parser.add_argument('-b', '--batch-size', dest='batch_size', type=int, default=512)
    parser.add_argument('-d', '--dense-size', dest='dense_size', type=int, default=256)
    parser.add_argument('-D', '--down-factor', dest='dfactor', type=int, default=50)
    parser.add_argument('-n', '--num-filters', action='store', dest='n_filters', nargs='+', type=int, default=[32])
    parser.add_argument('-e', '--epochs', action='store', dest='epochs', type=int, default=10)
    parser.add_argument('-E', '--epochs-repetition', action='store', dest='r_epochs', type=int, default=5)
    parser.add_argument('-q', '--queue', action='store', dest='queue', type=int, default=10)
    parser.add_argument('-v', '--validation-rate', action='store', dest='val_rate', type=float, default=0.25)
    parser.add_argument('-u', '--unbalanced', action='store_false', dest='balanced', default=True)
    parser.add_argument('-s', '--sequential', action='store_true', dest='sequential', default=False)
    parser.add_argument('-r', '--recurrent', action='store_true', dest='recurrent', default=False)
    parser.add_argument('-p', '--preload', action='store_true', dest='preload', default=False)
    parser.add_argument('-P', '--patience', dest='patience', type=int, default=5)
    parser.add_argument('--flair', action='store', dest='flair', default='_flair.nii.gz')
    parser.add_argument('--t1', action='store', dest='t1', default='_t1.nii.gz')
    parser.add_argument('--t1ce', action='store', dest='t1ce', default='_t1ce.nii.gz')
    parser.add_argument('--t2', action='store', dest='t2', default='_t2.nii.gz')
    parser.add_argument('--labels', action='store', dest='labels', default='_seg.nii.gz')
    return vars(parser.parse_args())


def train_net(net, checkpoint, val_layer_name, weights, nlabels):
    options = parse_inputs()
    c = color_codes()
    # Data stuff
    path = options['dir_name']
    train_data, train_labels = get_names_from_path(options)
    # Prepare the net architecture parameters
    dfactor = options['dfactor']
    # Prepare the net hyperparameters
    epochs = options['epochs']
    patch_width = options['patch_width']
    patch_size = (patch_width, patch_width, patch_width)
    batch_size = options['batch_size']
    conv_blocks = options['conv_blocks']
    conv_width = options['conv_width']
    kernel_size_list = conv_width if isinstance(conv_width, list) else [conv_width] * conv_blocks
    balanced = options['balanced']
    val_rate = options['val_rate']
    preload = options['preload']
    fc_width = patch_width - sum(kernel_size_list) + conv_blocks
    fc_shape = (fc_width,) * 3

    try:
        net = load_model(checkpoint)
    except IOError:

        callbacks = [
            EarlyStopping(monitor=val_layer_name, patience=options['patience']),
            ModelCheckpoint(os.path.join(path, checkpoint), monitor=val_layer_name, save_best_only=True)
        ]

        net.compile(optimizer='adadelta', loss='categorical_crossentropy', loss_weights=weights, metrics=['accuracy'])

        train_centers = get_cnn_centers(train_data[:, 0], train_labels, balanced=balanced)
        print(' '.join([''] * 16) + c['g'] + 'Loading data ' +
              c['b'] + '(%d centers)' % (len(train_centers) / dfactor) + c['nc'])
        x, y = load_patches_train(
            image_names=train_data,
            label_names=train_labels,
            centers=train_centers,
            size=patch_size,
            fc_shape=fc_shape,
            nlabels=nlabels,
            dfactor=dfactor,
            preload=preload,
            split=True,
            iseg=False,
            experimental=1,
            datatype=np.float32
        )

        print(' '.join([''] * 16) + c['g'] + 'Training the model for ' +
              c['b'] + '(%d parameters)' % net.count_params() + c['nc'])

        net.fit(x, y, batch_size=batch_size, validation_split=val_rate, epochs=epochs, callbacks=callbacks)
        net = load_model(checkpoint)

    return net


def train_net_loop(net, net_name, nlabels):
    options = parse_inputs()
    # Prepare the net hyperparameters
    r_epochs = options['r_epochs']
    c = color_codes()

    for i in range(r_epochs):
        val_loss = 'val_fc_out_loss'
        checkpoint = net_name + 'fc_out.e%d.best.hdf5' % i
        print('Epoch (' + c['b'] + val_loss + c['nc'] + ') %d/%d ' % (i + 1, r_epochs) + c['nc'])
        net = train_net(net, checkpoint, val_loss, [0.5, 1.0], nlabels)

        val_loss = 'val_tumor_loss'
        checkpoint = net_name + 'tumor.e%d.best.hdf5' % i
        print('Epoch (' + c['b'] + val_loss + c['nc'] + ') %d/%d ' % (i + 1, r_epochs) + c['nc'])
        net = train_net(net, checkpoint, val_loss, [1.0, 0.5], nlabels)


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
    epochs = options['epochs']
    r_epochs = options['r_epochs']
    patch_width = options['patch_width']
    patch_size = (patch_width, patch_width, patch_width)
    dense_size = options['dense_size']
    conv_blocks = options['conv_blocks']
    n_filters = options['n_filters']
    filters_list = n_filters if len(n_filters) > 1 else n_filters*conv_blocks
    conv_width = options['conv_width']
    kernel_size_list = conv_width if isinstance(conv_width, list) else [conv_width]*conv_blocks
    balanced = options['balanced']
    # Data loading parameters
    preload = options['preload']

    # Prepare the sufix that will be added to the results for the net and images
    path = options['dir_name']
    filters_s = 'n'.join(['%d' % nf for nf in filters_list])
    conv_s = 'c'.join(['%d' % cs for cs in kernel_size_list])
    ub_s = '.ub' if not balanced else ''
    params_s = (ub_s, dfactor, patch_width, conv_s, filters_s, dense_size, epochs, r_epochs)
    sufix = '%s.D%d.p%d.c%s.n%s.d%d.e%d.E%d.' % params_s
    preload_s = ' (with ' + c['b'] + 'preloading' + c['nc'] + c['c'] + ')' if preload else ''

    print(c['c'] + '[' + strftime("%H:%M:%S") + '] ' + 'Starting training' + preload_s + c['nc'])
    # N-fold cross validation main loop (we'll do 2 training iterations with testing for each patient)
    train_data, train_labels = get_names_from_path(options)

    print(c['c'] + '[' + strftime("%H:%M:%S") + ']  ' + c['nc'] + c['g'] +
          'Number of training images (%d=%d)' % (len(train_data), len(train_labels)) + c['nc'])
    #  Also, prepare the network

    print(c['c'] + '[' + strftime("%H:%M:%S") + ']    ' + c['g'] + 'Creating and compiling the model ' + c['nc'])
    input_shape = (train_data.shape[1],) + patch_size

    # Region based net
    roi_net = get_brats_net(input_shape, filters_list, kernel_size_list, dense_size, 2)
    roi_net_name = os.path.join(path, 'brats2017-roi' + sufix)
    train_net_loop(roi_net, roi_net_name, 2)

    seg_net = get_brats_net(input_shape, filters_list, kernel_size_list, dense_size, 5)

    # Tumor substrctures net
    roi_net_conv_layers = [l for l in roi_net.layers if 'conv' in l.name]
    seg_net_conv_layers = [l for l in seg_net.layers if 'conv' in l.name]
    for lr, ls in zip(roi_net_conv_layers[:conv_blocks], seg_net_conv_layers[:conv_blocks]):
        ls.set_weights(lr.get_weights())

    seg_net_name = os.path.join(path, 'brats2017-seg' + sufix)
    train_net_loop(seg_net, seg_net_name, 5)


if __name__ == '__main__':
    main()
