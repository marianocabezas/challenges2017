from __future__ import print_function
import argparse
import os
import sys
from time import strftime
import numpy as np
from nibabel import load as load_nii
from utils import color_codes
from data_creation import get_cnn_centers, load_norm_list, get_patches_list
from data_creation import load_patches_ganseg_by_batches, load_patches_gandisc_by_batches
from data_manipulation.generate_features import get_mask_voxels
from data_manipulation.metrics import dsc_seg
from nets import get_wmh_nets
import keras.backend as K


def parse_inputs():
    # I decided to separate this function, for easier acces to the command line parameters
    parser = argparse.ArgumentParser(description='Test different nets with 3D data.')
    parser.add_argument('-f', '--training-folder', dest='dir_train', default='/home/mariano/DATA/WMHTrain/')
    parser.add_argument('-F', '--test-folder', dest='dir_test', default='/home/mariano/DATA/WMHTest/')
    parser.add_argument('-i', '--patch-width', dest='patch_width', type=int, default=15)
    parser.add_argument('-k', '--kernel-size', dest='conv_width', nargs='+', type=int, default=3)
    parser.add_argument('-c', '--conv-blocks', dest='conv_blocks', type=int, default=4)
    parser.add_argument('-b', '--batch-size', dest='batch_size', type=int, default=128)
    parser.add_argument('-B', '--batch-test-size', dest='test_size', type=int, default=32768)
    parser.add_argument('-d', '--dense-size', dest='dense_size', type=int, default=256)
    parser.add_argument('-D', '--down-sampling', dest='downsample', type=int, default=1)
    parser.add_argument('-n', '--num-filters', action='store', dest='n_filters', nargs='+', type=int, default=[32])
    parser.add_argument('-e', '--epochs', action='store', dest='epochs', type=int, default=5)
    parser.add_argument('-v', '--validation-rate', action='store', dest='val_rate', type=float, default=0.25)
    parser.add_argument('-u', '--unbalanced', action='store_false', dest='balanced', default=True)
    parser.add_argument('-p', '--preload', action='store_true', dest='preload', default=False)
    parser.add_argument('-P', '--patience', dest='patience', type=int, default=2)
    parser.add_argument('--flair', action='store', dest='flair', default='pre/FLAIR.nii.gz')
    parser.add_argument('--t1', action='store', dest='t1', default='pre/T1.nii.gz')
    parser.add_argument('--labels', action='store', dest='labels', default='wmh.nii.gz')
    return vars(parser.parse_args())


def get_names_from_path(options, train=True):
    path = options['dir_train'] if train else options['dir_test']

    directories = filter(os.path.isdir, [os.path.join(path, f) for f in os.listdir(path)])
    patients = sorted(directories)

    # Prepare the names
    flair_names = [os.path.join(path, p, options['flair']) for p in patients]
    t1_names = [os.path.join(path, p, options['t1']) for p in patients]

    label_names = np.array([os.path.join(path, p, options['labels']) for p in patients])
    image_names = np.stack(filter(None, [flair_names, t1_names]), axis=1)

    return image_names, label_names


def train_nets(gan, cnn, p, x, y, name, adversarial_w, val_layer_name='val_loss'):
    options = parse_inputs()
    c = color_codes()
    # Data stuff
    patient_path = '/'.join(p[0].rsplit('/')[:-1])
    train_data, train_labels = get_names_from_path(options)
    # Prepare the net hyperparameters
    epochs = options['epochs']
    patch_width = options['patch_width']
    patch_size = (patch_width, patch_width, patch_width)
    preload = options['preload']
    batch_size = options['batch_size']
    val_rate = options['val_rate']

    print(c['c'] + '[' + strftime("%H:%M:%S") + ']    ' + c['g'] + 'Training the networks ' + c['nc'] +
          '(' + c['y'] + 'GAN' + c['nc'] + '/' + c['lgy'] + 'CNN' + c['nc'] + ': ' +
          c['b'] + '%d' % gan.count_params() + c['nc'] + '/' + c['b'] + '%d ' % cnn.count_params() + c['nc'] +
          'parameters)')

    net_name = os.path.join(patient_path, name)
    checkpoint_name = os.path.join(patient_path, net_name + '.weights')

    try:
        gan.load_weights(checkpoint_name + '.gan.e%d' % epochs)
        cnn.load_weights(checkpoint_name + '.net.e%d' % epochs)
    except IOError:
        x_disc, y_disc = load_patches_gandisc_by_batches(
            source_names=train_data,
            target_names=[p],
            n_centers=len(x),
            size=patch_size,
            preload=preload,
        )
        print(' '.join([''] * 15) + c['g'] + 'Starting the training process' + c['nc'])
        for e in range(epochs):
            print(' '.join([''] * 16) + c['g'] + 'Epoch ' +
                  c['b'] + '%d' % (e + 1) + c['nc'] + c['g'] + '/%d' % epochs + c['nc'])
            try:
                gan.load_weights(checkpoint_name + '.gan.e%d' % (e + 1))
                cnn.load_weights(checkpoint_name + '.net.e%d' % (e + 1))
            except IOError:
                print(c['y'], end='\r')
                gan.fit([x, x_disc], [y, y_disc], batch_size=batch_size, epochs=1)
                print(c['lgy'], end='\r')
                cnn.fit(x, y, batch_size=batch_size, epochs=1)
                print(c['nc'], end='\r')

            gan.save_weights(checkpoint_name + '.gan.e%d' % (e + 1))
            cnn.save_weights(checkpoint_name + '.net.e%d' % (e + 1))
            adversarial_w += 1.0 / (epochs - 1)


def test_net(net, p, outputname):

    c = color_codes()
    options = parse_inputs()
    patch_width = options['patch_width']
    patch_size = (patch_width, patch_width, patch_width)
    batch_size = options['test_size']
    p_name = p[0].rsplit('/')[-2]
    patient_path = '/'.join(p[0].rsplit('/')[:-1])
    outputname_path = os.path.join(patient_path, outputname + '.nii.gz')
    pr_outputname_path = os.path.join(patient_path, outputname + '.pr.nii.gz')
    try:
        image = load_nii(outputname_path).get_data()
    except IOError:
        print(c['c'] + '[' + strftime("%H:%M:%S") + ']    ' + c['g'] + 'Testing the network' + c['nc'])
        nii = load_nii(p[0])
        roi = nii.get_data().astype(dtype=np.bool)
        centers = get_mask_voxels(roi)
        test_samples = np.count_nonzero(roi)
        image = np.zeros_like(roi).astype(dtype=np.uint8)
        pr = np.zeros_like(roi).astype(dtype=np.float32)
        print(c['c'] + '[' + strftime("%H:%M:%S") + ']    ' + c['g'] +
              '<Creating the probability map ' + c['b'] + p_name + c['nc'] + c['g'] + ' - ' +
              c['b'] + outputname + c['nc'] + c['g'] + ' (%d samples)>' % test_samples + c['nc'])

        n_centers = len(centers)
        image_list = [load_norm_list(p)]

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

            # We store the results
            image[x, y, z] = np.argmax(y_pr_pred, axis=1).astype(dtype=np.int8)
            pr[x, y, z] = y_pr_pred[:, 1].astype(dtype=np.float32)

        print(' '.join([''] * 50), end='\r')
        sys.stdout.flush()

        # Post-processing (Basically keep the biggest connected region)
        # image = get_biggest_region(image)
        print(c['g'] + '                   -- Saving image ' + c['b'] + outputname_path + c['nc'])

        nii.get_data()[:] = image
        nii.to_filename(outputname_path)
        nii.get_data()[:] = pr
        nii.to_filename(pr_outputname_path)
    return image


def main():
    options = parse_inputs()
    c = color_codes()

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
    downsample = options['downsample']
    preload = options['preload']

    # Prepare the sufix that will be added to the results for the net and images
    filters_s = 'n'.join(['%d' % nf for nf in filters_list])
    conv_s = 'c'.join(['%d' % cs for cs in kernel_size_list])
    ub_s = '.ub' if not balanced else ''
    params_s = (ub_s, patch_width, conv_s, filters_s, dense_size, downsample)
    sufix = '%s.p%d.c%s.n%s.d%d.D%d' % params_s
    preload_s = ' (with ' + c['b'] + 'preloading' + c['nc'] + c['c'] + ')' if preload else ''

    print(c['c'] + '[' + strftime("%H:%M:%S") + '] ' + 'Starting training' + preload_s + c['nc'])
    train_data, _ = get_names_from_path(options)
    test_data, test_labels = get_names_from_path(options, False)

    input_shape = (train_data.shape[1],) + patch_size

    dsc_results = list()

    train_data, train_labels = get_names_from_path(options)
    centers_s = np.random.permutation(
        get_cnn_centers(train_data[:, 0], train_labels, balanced=balanced)
    )[::downsample]
    x_seg, y_seg = load_patches_ganseg_by_batches(
        image_names=train_data,
        label_names=train_labels,
        source_centers=centers_s,
        size=patch_size,
        nlabels=2,
        preload=preload,
    )

    for i, (p, gt_name) in enumerate(zip(test_data, test_labels)):
        p_name = p[0].rsplit('/')[-3]
        patient_path = '/'.join(p[0].rsplit('/')[:-1])
        print(c['c'] + '[' + strftime("%H:%M:%S") + ']  ' + c['nc'] + 'Case ' + c['c'] + c['b'] + p_name + c['nc'] +
              c['c'] + ' (%d/%d):' % (i + 1, len(test_data)) + c['nc'])

        # ROI segmentation
        adversarial_w = K.variable(0)
        cnn, gan, gan_test = get_wmh_nets(
            input_shape=input_shape,
            filters_list=filters_list,
            kernel_size_list=kernel_size_list,
            dense_size=dense_size,
            lambda_var=adversarial_w
        )
        train_nets(
            gan=gan,
            cnn=cnn,
            p=p,
            x=x_seg,
            y=y_seg,
            name='wmh2017' + sufix,
            adversarial_w=adversarial_w
        )

        image_cnn_name = os.path.join(patient_path, p_name + '.cnn.test.e%d' % epochs)
        try:
            image_cnn = load_nii(image_cnn_name + '.nii.gz').get_data()
        except IOError:
            image_cnn = test_net(cnn, p, image_cnn_name)
        seg_cnn = image_cnn.astype(np.bool)

        image_gan_name = os.path.join(patient_path, p_name + '.gan.test.e%d' % epochs)
        try:
            image_gan = load_nii(image_gan_name + '.nii.gz').get_data()
        except IOError:
            image_gan = test_net(gan_test, p, image_gan_name)
        seg_gan = image_gan.astype(np.bool)

        seg_gt = load_nii(gt_name).get_data().astype(np.bool)

        results_cnn = dsc_seg(seg_gt, seg_cnn)
        results_gan = dsc_seg(seg_gt, seg_gan)
        print(''.join([' '] * 14) + c['c'] + c['b'] + p_name + c['nc'] + ' ' + c['lgy'] + 'CNN' + c['nc'] +
              ' vs ' + c['y'] + 'GAN' + c['nc'] + ' DSC: %f vs %f' % (results_cnn, results_gan))

        dsc_results.append((results_cnn, results_gan))

    final_dsc_string = 'Final results DSC: ' + c['lgy'] + '%f' + c['nc'] + ' vs '  + c['y'] + '%f' + c['nc']
    print(final_dsc_string % tuple(np.mean(dsc_results, axis=0)))

if __name__ == '__main__':
    main()
