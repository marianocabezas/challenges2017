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
from nets import get_brats_gan_fc, get_brats_fc, get_brats_caps
import keras.backend as K


def parse_inputs():
    # I decided to separate this function, for easier acces to the command line parameters
    parser = argparse.ArgumentParser(description='Test different nets with 3D data.')
    parser.add_argument('-f', '--training-folder', dest='dir_train', default='/home/mariano/DATA/Brats17Train/')
    parser.add_argument('-F', '--test-folder', dest='dir_test', default='/home/mariano/DATA/Brats17Test/')
    parser.add_argument('-i', '--patch-width', dest='patch_width', type=int, default=25)
    parser.add_argument('-k', '--kernel-size', dest='conv_width', nargs='+', type=int, default=3)
    parser.add_argument('-c', '--conv-blocks', dest='conv_blocks', type=int, default=8)
    parser.add_argument('-b', '--batch-size', dest='batch_size', type=int, default=128)
    parser.add_argument('-B', '--batch-test-size', dest='test_size', type=int, default=32768)
    parser.add_argument('-d', '--dense-size', dest='dense_size', type=int, default=256)
    parser.add_argument('-s', '--down-sampling', dest='down_sampling', type=int, default=1)
    parser.add_argument('-n', '--num-filters', action='store', dest='n_filters', nargs='+', type=int, default=[32])
    parser.add_argument('-e', '--epochs', action='store', dest='epochs', type=int, default=5)
    parser.add_argument('-v', '--validation-rate', action='store', dest='val_rate', type=float, default=0.25)
    parser.add_argument('-u', '--unbalanced', action='store_false', dest='balanced', default=True)
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

    directories = filter(os.path.isdir, [os.path.join(path, f) for f in os.listdir(path)])
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


def train_nets(gan, cnn, caps, x, y, p, name, adversarial_w):
    options = parse_inputs()
    c = color_codes()
    # Data stuff
    patient_path = '/'.join(p[0].rsplit('/')[:-1])
    train_data, train_labels = get_names_from_path(options)
    # Prepare the net hyperparameters
    epochs = options['epochs']
    patch_width = options['patch_width']
    patch_size = (patch_width, patch_width, patch_width)
    batch_size = options['batch_size']
    preload = options['preload']

    print(c['c'] + '[' + strftime("%H:%M:%S") + ']    ' + c['g'] + 'Training the networks ' + c['nc'] +
          c['lgy'] + '(' + 'CNN' + c['nc'] + '/' +
          c['r'] + 'CAPS' + c['nc'] + '/' +
          c['y'] + 'GAN' + c['nc'] + ': ' +
          c['b'] + '%d' % gan.count_params() + c['nc'] + '/' + c['b'] + '%d ' % cnn.count_params() + c['nc'] +
          'parameters)')

    net_name = os.path.join(patient_path, name)
    checkpoint_name = os.path.join(patient_path, net_name + '.weights')

    try:
        cnn.load_weights(checkpoint_name + '.net.e%d' % epochs)
        caps.load_weights(checkpoint_name + '.caps.e%d' % epochs)
        gan.load_weights(checkpoint_name + '.gan.e%d' % epochs)
    except IOError:
        x_disc, y_disc = load_patches_gandisc_by_batches(
            source_names=train_data,
            target_names=[p],
            n_centers=len(x),
            size=patch_size,
            preload=preload,
            batch_size=51200
        )
        print(' '.join([''] * 15) + c['g'] + 'Starting the training process' + c['nc'])
        for e in range(epochs):
            print(' '.join([''] * 16) + c['g'] + 'Epoch ' +
                  c['b'] + '%d' % (e + 1) + c['nc'] + c['g'] + '/%d' % epochs + c['nc'])
            try:
                cnn.load_weights(checkpoint_name + '.net.e%d' % (e + 1))
            except IOError:
                print(c['lgy'], end='\r')
                cnn.fit(x, y, batch_size=batch_size, epochs=1)
            try:
                caps.load_weights(checkpoint_name + '.caps.e%d' % (e + 1))
            except IOError:
                print(c['r'], end='\r')
                caps.fit(x, y, batch_size=batch_size, epochs=1)
            try:
                gan.load_weights(checkpoint_name + '.gan.e%d' % (e + 1))
            except IOError:
                print(c['y'], end='\r')
                gan.fit([x, x_disc], [y, y_disc], batch_size=batch_size, epochs=1)
            print(c['nc'], end='\r')

            cnn.save_weights(checkpoint_name + '.net.e%d' % (e + 1))
            caps.save_weights(checkpoint_name + '.caps.e%d' % (e + 1))
            gan.save_weights(checkpoint_name + '.gan.e%d' % (e + 1))
            adversarial_weight = min([np.array(K.eval(adversarial_w)) + 0.1, 1.0])
            K.set_value(adversarial_w, adversarial_weight)


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
            roi[x, y, z] = np.argmax(y_pr_pred, axis=1).astype(dtype=np.bool)
            # We store the results
            image[x, y, z] = np.argmax(y_pr_pred, axis=1).astype(dtype=np.int8)

        print(' '.join([''] * 50), end='\r')
        sys.stdout.flush()

        # Post-processing (Basically keep the biggest connected region)
        # image = get_biggest_region(image)
        print(c['g'] + '                   -- Saving image ' + c['b'] + outputname_path + c['nc'])

        roi_nii.get_data()[:] = roi
        roi_nii.to_filename(roiname)

        roi_nii.get_data()[:] = image
        roi_nii.to_filename(outputname_path)
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
    preload = options['preload']

    # Prepare the sufix that will be added to the results for the net and images
    filters_s = 'n'.join(['%d' % nf for nf in filters_list])
    conv_s = 'c'.join(['%d' % cs for cs in kernel_size_list])
    ub_s = '.ub' if not balanced else ''
    params_s = (ub_s, patch_width, conv_s, filters_s, dense_size, epochs)
    sufix = '%s.p%d.c%s.n%s.d%d.e%d' % params_s
    preload_s = ' (with ' + c['b'] + 'preloading' + c['nc'] + c['c'] + ')' if preload else ''

    print(c['c'] + '[' + strftime("%H:%M:%S") + '] ' + 'Starting training' + preload_s + c['nc'])
    train_data, _ = get_names_from_path(options)
    test_data, test_labels = get_names_from_path(options, False)

    input_shape = (train_data.shape[1],) + patch_size

    dsc_results_gan = list()
    dsc_results_cnn = list()
    dsc_results_caps = list()

    train_data, train_labels = get_names_from_path(options)
    centers_s = np.random.permutation(
        get_cnn_centers(train_data[:, 0], train_labels, balanced=balanced)
    )[::options['down_sampling']]
    x_seg, y_seg = load_patches_ganseg_by_batches(
        image_names=train_data,
        label_names=train_labels,
        source_centers=centers_s,
        size=patch_size,
        nlabels=5,
        preload=preload,
        batch_size=51200
    )

    y_seg_roi = np.empty((len(y_seg), 2), dtype=np.bool)
    y_seg_roi[:, 0] = y_seg[:, 0]
    y_seg_roi[:, 1] = np.sum(y_seg[:, 1:], axis=1)

    for i, (p, gt_name) in enumerate(zip(test_data, test_labels)):
        p_name = p[0].rsplit('/')[-2]
        patient_path = '/'.join(p[0].rsplit('/')[:-1])
        print('%s[%s] %sCase %s%s%s%s%s (%d/%d):%s' % (
            c['c'],
            strftime("%H:%M:%S"),
            c['nc'],
            c['c'],
            c['b'],
            p_name,
            c['nc'],
            c['c'],
            i + 1,
            len(test_data),
            c['nc']
        ))

        # ROI segmentation
        adversarial_w = K.variable(0)
        roi_cnn = get_brats_fc(input_shape, filters_list, kernel_size_list, dense_size, 2)
        roi_caps = get_brats_caps(input_shape, filters_list, kernel_size_list, 8, 2)
        roi_gan, _ = get_brats_gan_fc(
            input_shape,
            filters_list,
            kernel_size_list,
            dense_size,
            2,
            lambda_var=adversarial_w
        )
        train_nets(
            x=x_seg,
            y=y_seg_roi,
            gan=roi_gan,
            cnn=roi_cnn,
            caps=roi_caps,
            p=p,
            name='brats2017-roi' + sufix,
            adversarial_w=adversarial_w
        )

        # Tumor substructures net
        adversarial_w = K.variable(0)
        seg_cnn = get_brats_fc(input_shape, filters_list, kernel_size_list, dense_size, 5)
        seg_caps = get_brats_caps(input_shape, filters_list, kernel_size_list, 8, 5)
        seg_gan_tr, seg_gan_tst = get_brats_gan_fc(
            input_shape,
            filters_list,
            kernel_size_list,
            dense_size,
            5,
            lambda_var=adversarial_w
        )
        roi_net_conv_layers = [l for l in roi_gan.layers if 'conv' in l.name]
        seg_net_conv_layers = [l for l in seg_gan_tr.layers if 'conv' in l.name]
        for lr, ls in zip(roi_net_conv_layers[:conv_blocks], seg_net_conv_layers[:conv_blocks]):
            ls.set_weights(lr.get_weights())
        train_nets(
            x=x_seg,
            y=y_seg,
            gan=seg_gan_tr,
            cnn=seg_cnn,
            caps=seg_caps,
            p=p,
            name='brats2017-full' + sufix,
            adversarial_w=adversarial_w
        )

        image_cnn_name = os.path.join(patient_path, p_name + '.cnn.test')
        try:
            image_cnn = load_nii(image_cnn_name + '.nii.gz').get_data()
        except IOError:
            image_cnn = test_net(seg_cnn, p, image_cnn_name)

        image_caps_name = os.path.join(patient_path, p_name + '.caps.test')
        try:
            image_caps = load_nii(image_caps_name + '.nii.gz').get_data()
        except IOError:
            image_caps = test_net(seg_caps, p, image_caps_name)

        image_gan_name = os.path.join(patient_path, p_name + '.gan.test')
        try:
            image_gan = load_nii(image_gan_name + '.nii.gz').get_data()
        except IOError:
            image_gan = test_net(seg_gan_tst, p, image_gan_name)

        results_cnn = check_dsc(gt_name, image_cnn)
        dsc_string = c['g'] + '/'.join(['%f'] * len(results_cnn)) + c['nc']
        print(''.join([' '] * 14) + c['c'] + c['b'] + p_name + c['nc'] + ' CNN DSC: ' +
              dsc_string % tuple(results_cnn))

        results_caps = check_dsc(gt_name, image_caps)
        dsc_string = c['g'] + '/'.join(['%f'] * len(results_caps)) + c['nc']
        print(''.join([' '] * 14) + c['c'] + c['b'] + p_name + c['nc'] + ' CAPS DSC: ' +
              dsc_string % tuple(results_caps))

        results_gan = check_dsc(gt_name, image_gan)
        dsc_string = c['g'] + '/'.join(['%f'] * len(results_gan)) + c['nc']
        print(''.join([' '] * 14) + c['c'] + c['b'] + p_name + c['nc'] + ' GAN DSC: ' +
              dsc_string % tuple(results_gan))

        dsc_results_cnn.append(results_cnn)
        dsc_results_caps.append(results_caps)
        dsc_results_gan.append(results_gan)

    f_dsc = tuple(
        [np.array([dsc[i] for dsc in dsc_results_cnn if len(dsc) > i]).mean() for i in range(3)]
    ) + tuple(
        [np.array([dsc[i] for dsc in dsc_results_caps if len(dsc) > i]).mean() for i in range(3)]
    ) + tuple(
        [np.array([dsc[i] for dsc in dsc_results_gan if len(dsc) > i]).mean() for i in range(3)]
    )
    print('Final results DSC: (%f/%f/%f) vs (%f/%f/%f) vs (%f/%f/%f)' % f_dsc)


if __name__ == '__main__':
    main()
