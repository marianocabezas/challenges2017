from __future__ import print_function
import argparse
import os
import sys
from time import strftime
import numpy as np
from nibabel import load as load_nii
from utils import color_codes, get_biggest_region
from data_creation import get_cnn_centers, load_norm_list, get_patches_list, load_patches_gan
from data_manipulation.generate_features import get_mask_voxels
from data_manipulation.metrics import dsc_seg
from nets import get_brats_gan
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
import keras.backend as K


def parse_inputs():
    # I decided to separate this function, for easier acces to the command line parameters
    parser = argparse.ArgumentParser(description='Test different nets with 3D data.')
    parser.add_argument('-f', '--training-folder', dest='dir_train', default='/home/mariano/DATA/Brats17Train/')
    parser.add_argument('-F', '--test-folder', dest='dir_test', default='/home/mariano/DATA/Brats17Test/')
    parser.add_argument('-i', '--patch-width', dest='patch_width', type=int, default=17)
    parser.add_argument('-k', '--kernel-size', dest='conv_width', nargs='+', type=int, default=3)
    parser.add_argument('-c', '--conv-blocks', dest='conv_blocks', type=int, default=4)
    parser.add_argument('-b', '--batch-size', dest='batch_size', type=int, default=512)
    parser.add_argument('-B', '--batch-test-size', dest='test_size', type=int, default=32768)
    parser.add_argument('-d', '--dense-size', dest='dense_size', type=int, default=256)
    parser.add_argument('-D', '--down-factor', dest='dfactor', type=int, default=50)
    parser.add_argument('-n', '--num-filters', action='store', dest='n_filters', nargs='+', type=int, default=[32])
    parser.add_argument('-e', '--epochs', action='store', dest='epochs', type=int, default=5)
    parser.add_argument('-q', '--queue', action='store', dest='queue', type=int, default=10)
    parser.add_argument('-v', '--validation-rate', action='store', dest='val_rate', type=float, default=0.25)
    parser.add_argument('-u', '--unbalanced', action='store_false', dest='balanced', default=True)
    parser.add_argument('-s', '--sequential', action='store_true', dest='sequential', default=False)
    parser.add_argument('-r', '--recurrent', action='store_true', dest='recurrent', default=False)
    parser.add_argument('-p', '--preload', action='store_true', dest='preload', default=False)
    parser.add_argument('-P', '--patience', dest='patience', type=int, default=2)
    parser.add_argument('--flair', action='store', dest='flair', default='_flair.nii.gz')
    parser.add_argument('--t1', action='store', dest='t1', default='_t1.nii.gz')
    parser.add_argument('--t1ce', action='store', dest='t1ce', default='_t1ce.nii.gz')
    parser.add_argument('--t2', action='store', dest='t2', default='_t2.nii.gz')
    parser.add_argument('--labels', action='store', dest='labels', default='_seg.nii.gz')
    return vars(parser.parse_args())


def get_names_from_path(options, train=True):
    path = options['dir_train'] if train else options['dir_test']

    directories= filter(os.path.isdir, [os.path.join(path, f) for f in os.listdir(path)])
    patients = sorted(directories)

    # Prepare the names
    flair_names = [os.path.join(path, p, p.split('/')[-1] + options['flair']) for p in patients]
    t2_names = [os.path.join(path, p, p.split('/')[-1] + options['t2']) for p in patients]
    t1_names = [os.path.join(path, p, p.split('/')[-1] + options['t1']) for p in patients]
    t1ce_names = [os.path.join(path, p, p.split('/')[-1] + options['t1ce']) for p in patients]

    label_names = np.array([os.path.join(path, p, p.split('/')[-1] + options['labels']) for p in patients])
    image_names = np.stack(filter(None, [flair_names, t2_names, t1_names, t1ce_names]), axis=1)

    return image_names, label_names


def check_dsc(gt_name, image):
    gt_nii = load_nii(gt_name)
    gt = np.copy(gt_nii.get_data()).astype(dtype=np.uint8)
    labels = np.unique(gt.flatten())
    return [dsc_seg(gt == l, image == l) for l in labels[1:]]


def train_net(net, p, name, val_layer_name='val_loss', nlabels=5):
    options = parse_inputs()
    c = color_codes()
    # Data stuff
    patient_path = '/'.join(p[0].rsplit('/')[:-1])
    train_data, train_labels = get_names_from_path(options)
    # Prepare the net architecture parameters
    dfactor = options['dfactor']
    # Prepare the net hyperparameters
    epochs = options['epochs']
    patch_width = options['patch_width']
    patch_size = (patch_width, patch_width, patch_width)
    batch_size = options['batch_size']
    balanced = options['balanced']
    val_rate = options['val_rate']
    preload = options['preload']

    adversarial_w = K.variable(0)

    net_name = os.path.join(patient_path, name)
    checkpoint_name = os.path.join(patient_path, net_name + '.weights')

    try:
        net = load_model(net_name)
    except IOError:
        net.save(net_name)
        centers_s = np.random.permutation(get_cnn_centers(train_data[:, 0], train_labels, balanced=balanced))
        print(' '.join([''] * 15) + c['g'] + 'Total number of centers = ' +
              c['b'] + '(%d centers)' % (len(centers_s)) + c['nc'])
        for i in range(dfactor):
            print(' '.join([''] * 16) + c['g'] + 'Round ' +
                  c['b'] + '%d' % (i + 1) + c['nc'] + c['g'] + '/%d' % dfactor + c['nc'])
            try:
                net.load_weights(checkpoint_name + '.e%d' % (i+1))
            except IOError:
                batch_centers_s = centers_s[i::dfactor]
                print(' '.join([''] * 16) + c['g'] + 'Loading data ' +
                      c['b'] + '(%d centers)' % (len(batch_centers_s) * 2) + c['nc'])
                x, y = load_patches_gan(
                    source_names=train_data,
                    target_names=[p],
                    label_names=train_labels,
                    source_centers=batch_centers_s,
                    size=patch_size,
                    nlabels=nlabels,
                    preload=preload,
                )

                print(' '.join([''] * 16) + c['g'] + 'Training the model for ' +
                      c['b'] + '(%d parameters)' % net.count_params() + c['nc'])
                net.compile(
                    optimizer='adadelta',
                    loss={'seg': 'categorical_crossentropy', 'disc': 'binary_crossentropy'},
                    loss_weights=[1, adversarial_w],
                    metrics=['accuracy']
                )

                adversarial_w -= 1.0 / dfactor

                callbacks = [
                    EarlyStopping(
                        monitor=val_layer_name,
                        patience=options['patience']
                    ),
                    ModelCheckpoint(
                        checkpoint_name + '.e%d' % (i+1),
                        monitor=val_layer_name,
                        save_best_only=True,
                        save_weights_only=True
                    )
                ]

                net.fit(x, y, batch_size=batch_size, validation_split=val_rate, epochs=epochs, callbacks=callbacks)
                net.load_weights(checkpoint_name + '.e%d' % (i+1))


def test_net(net, p, outputname):

    c = color_codes()
    options = parse_inputs()
    patch_width = options['patch_width']
    patch_size = (patch_width, patch_width, patch_width)
    batch_size = options['test_size']
    p_name = p[0].rsplit('/')[-2]
    patient_path = '/'.join(p[0].rsplit('/')[:-1])
    outputname_path = os.path.join(patient_path, outputname + '.nii.gz')
    roiname = os.path.join(patient_path, outputname + '.roi.nii.gz')
    try:
        image = load_nii(outputname_path).get_data()
        load_nii(roiname)
    except IOError:
        print(c['c'] + '[' + strftime("%H:%M:%S") + ']    ' + c['g'] + 'Testing the network' + c['nc'])
        roi_nii = load_nii(p[0])
        roi = roi_nii.get_data().astype(dtype=np.bool)
        centers = get_mask_voxels(roi)
        test_samples = np.count_nonzero(roi)
        image = np.zeros_like(roi).astype(dtype=np.uint8)
        print(c['c'] + '[' + strftime("%H:%M:%S") + ']    ' + c['g'] +
              '<Creating the probability map ' + c['b'] + p_name + c['nc'] + c['g'] + ' - ' +
              c['b'] + outputname + c['nc'] + c['g'] + ' (%d samples)>' % test_samples + c['nc'])

        n_centers = len(centers)
        image_list = [load_norm_list(p)]
        is_roi = True
        roi = np.zeros_like(roi).astype(dtype=np.uint8)

        for i in range(0, n_centers, batch_size):
            print(
                '%f%% tested (step %d/%d)' % (100.0 * i / n_centers, (i / batch_size) + 1, -(-n_centers/batch_size)),
                end='\r'
            )
            sys.stdout.flush()
            centers_i = [centers[i:i + batch_size]]
            x = get_patches_list(image_list, centers_i, patch_size, True)
            x = np.concatenate(x).astype(dtype=np.float32)
            y_pr_pred = net.predict(x, batch_size=options['batch_size'])

            [x, y, z] = np.stack(centers_i[0], axis=1)

            # We store the ROI
            roi[x, y, z] = np.argmax(y_pr_pred[0], axis=1).astype(dtype=np.bool)
            # We store the results
            image[x, y, z] = np.argmax(y_pr_pred[0], axis=1).astype(dtype=np.int8)

        print(' '.join([''] * 50), end='\r')
        sys.stdout.flush()

        # Post-processing (Basically keep the biggest connected region)
        image = get_biggest_region(image, is_roi)
        print(c['g'] + '                   -- Saving image ' + c['b'] + outputname_path + c['nc'])

        roi_nii.get_data()[:] = roi
        roi_nii.to_filename(roiname)

        roi_nii.get_data()[:] = image
        roi_nii.to_filename(outputname_path)
    return image


def main():
    options = parse_inputs()
    c = color_codes()

    # Prepare the net architecture parameters
    dfactor = options['dfactor']
    # Prepare the net hyperparameters
    epochs = options['epochs']
    patch_width = options['patch_width']
    patch_size = (patch_width, patch_width, patch_width)
    dense_size = options['dense_size']
    conv_blocks = options['conv_blocks']
    n_filters = options['n_filters']
    filters_list = n_filters if len(n_filters) > 1 else n_filters * conv_blocks
    conv_width = options['conv_width']
    kernel_size_list = conv_width if isinstance(conv_width, list) else [conv_width] * conv_blocks
    balanced = options['balanced']
    # Data loading parameters
    preload = options['preload']

    # Prepare the sufix that will be added to the results for the net and images
    path = options['dir_train']
    filters_s = 'n'.join(['%d' % nf for nf in filters_list])
    conv_s = 'c'.join(['%d' % cs for cs in kernel_size_list])
    ub_s = '.ub' if not balanced else ''
    params_s = (ub_s, dfactor, patch_width, conv_s, filters_s, dense_size, epochs)
    sufix = '%s.D%d.p%d.c%s.n%s.d%d.e%d' % params_s
    preload_s = ' (with ' + c['b'] + 'preloading' + c['nc'] + c['c'] + ')' if preload else ''

    print(c['c'] + '[' + strftime("%H:%M:%S") + '] ' + 'Starting training' + preload_s + c['nc'])
    train_data, _ = get_names_from_path(options)
    test_data, test_labels = get_names_from_path(options, False)

    input_shape = (train_data.shape[1],) + patch_size

    for i, (p, gt_name) in enumerate(zip(test_data, test_labels)):
        p_name = p[0].rsplit('/')[-2]
        print(c['c'] + '[' + strftime("%H:%M:%S") + ']  ' + c['nc'] + 'Case ' + c['c'] + c['b'] + p_name + c['nc'] +
              c['c'] + ' (%d/%d):' % (i + 1, len(test_data)) + c['nc'])
        roi_net = get_brats_gan(input_shape, filters_list, kernel_size_list, dense_size, 2)
        train_net(roi_net, p, 'brats2017-roi' + sufix, nlabels=2)

        seg_net = get_brats_gan(input_shape, filters_list, kernel_size_list, dense_size, 5)
        # Tumor substrctures net
        roi_net_conv_layers = [l for l in roi_net.layers if 'conv' in l.name]
        seg_net_conv_layers = [l for l in seg_net.layers if 'conv' in l.name]
        for lr, ls in zip(roi_net_conv_layers[:conv_blocks], seg_net_conv_layers[:conv_blocks]):
            ls.set_weights(lr.get_weights())
        train_net(seg_net, p, 'brats2017-full' + sufix, nlabels=5)


if __name__ == '__main__':
    main()
