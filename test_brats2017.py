from __future__ import print_function
import argparse
import pickle
import os
import sys
from itertools import product
from time import strftime
import numpy as np
import keras
import keras.backend as K
from keras.models import Model
from keras.layers import Conv3D, Dropout, Input, Reshape, Lambda, Dense
from nibabel import load as load_nii
from utils import color_codes, get_biggest_region
from data_creation import load_norm_list, clip_to_roi
from data_creation import get_patches_list
from data_manipulation.generate_features import get_mask_voxels, get_patches
from data_manipulation.metrics import dsc_seg
from nets import get_brats_net
from scipy.ndimage.interpolation import zoom
from skimage.measure import compare_ssim as ssim


def check_dsc(gt_name, image):
    gt_nii = load_nii(gt_name)
    gt = np.copy(gt_nii.get_data()).astype(dtype=np.uint8)
    labels = np.unique(gt.flatten())
    return [dsc_seg(gt == l, image == l) for l in labels[1:]]


def parse_inputs():
    # I decided to separate this function, for easier acces to the command line parameters
    parser = argparse.ArgumentParser(description='Test different nets with 3D data.')
    parser.add_argument('-f', '--folder', dest='dir_name', default='/home/mariano/DATA/Brats17Test/')
    parser.add_argument('-i', '--patch-width', dest='patch_width', type=int, default=13)
    parser.add_argument('-k', '--kernel-size', dest='conv_width', nargs='+', type=int, default=3)
    parser.add_argument('-c', '--conv-blocks', dest='conv_blocks', type=int, default=5)
    parser.add_argument('-b', '--batch-size', dest='batch_size', type=int, default=2048)
    parser.add_argument('-D', '--down-factor', dest='down_factor', type=int, default=4)
    parser.add_argument('-n', '--num-filters', action='store', dest='n_filters', nargs='+', type=int, default=[32])
    parser.add_argument('-e', '--epochs', action='store', dest='epochs', type=int, default=2)
    parser.add_argument('-E', '--net-epochs', action='store', dest='net_epochs', type=int, default=1)
    parser.add_argument('--no-flair', action='store_false', dest='use_flair', default=True)
    parser.add_argument('--no-t1', action='store_false', dest='use_t1', default=True)
    parser.add_argument('--no-t1ce', action='store_false', dest='use_t1ce', default=True)
    parser.add_argument('--no-t2', action='store_false', dest='use_t2', default=True)
    parser.add_argument('--no-dsc', action='store_false', dest='use_dsc', default=True)
    parser.add_argument('--flair', action='store', dest='flair', default='_flair.nii.gz')
    parser.add_argument('--t1', action='store', dest='t1', default='_t1.nii.gz')
    parser.add_argument('--t1ce', action='store', dest='t1ce', default='_t1ce.nii.gz')
    parser.add_argument('--t2', action='store', dest='t2', default='_t2.nii.gz')
    parser.add_argument('--labels', action='store', dest='labels', default='_seg.nii.gz')
    return vars(parser.parse_args())


def list_directories(path):
    return filter(os.path.isdir, [os.path.join(path, f) for f in os.listdir(path)])


def get_names_from_path(path, options):
    patients = sorted(list_directories(path))

    # Prepare the names
    flair_names = [os.path.join(path, p, p.split('/')[-1] + options['flair'])
                   for p in patients]
    t2_names = [os.path.join(path, p, p.split('/')[-1] + options['t2'])
                for p in patients]
    t1_names = [os.path.join(path, p, p.split('/')[-1] + options['t1'])
                for p in patients]
    t1ce_names = [os.path.join(path, p, p.split('/')[-1] + options['t1ce'])
                  for p in patients]
    label_names = np.array([os.path.join(path, p, p.split('/')[-1] + options['labels']) for p in patients])
    image_names = np.stack(filter(None, [flair_names, t2_names, t1_names, t1ce_names]), axis=1)

    return image_names, label_names


def data_loading(image, labels, centers, fcn=False):
    c = color_codes()
    options = parse_inputs()
    # Network hyperparameters
    conv_blocks = options['conv_blocks']
    conv_width = options['conv_width']
    kernel_size_list = conv_width if isinstance(conv_width, list) else [conv_width] * conv_blocks
    patch_width = options['patch_width'] if not fcn else 17
    patch_size = (patch_width, patch_width, patch_width)
    d_factor = options['down_factor']

    centers = [tuple(center) for center in np.random.permutation(centers)[::d_factor]]
    print(c['c'] + '[' + strftime("%H:%M:%S") + ']    ' + c['g'] + 'Preparing ' + c['b'] + 'net' + c['nc'] +
          c['g'] + ' data (' + c['b'] + '%d' % len(centers) + c['nc'] + c['g'] + ' samples)' + c['nc'])
    x = [get_patches(image, centers, patch_size)
         for image in image]
    x = np.stack(x, axis=1).astype(np.float32)
    y = np.array([labels[center] for center in centers])
    if not fcn:
        y = [
            keras.utils.to_categorical(
                np.copy(y).astype(dtype=np.bool),
                num_classes=2
            ),
            keras.utils.to_categorical(
                np.array(y > 0).astype(dtype=np.int8) + np.array(y > 1).astype(dtype=np.int8),
                num_classes=3
            ),
            keras.utils.to_categorical(
                y,
                num_classes=5
            )
        ]
    else:
        fc_width = patch_width - sum(kernel_size_list) + conv_blocks
        fc_shape = (fc_width,) * 3
        y_fc = [np.asarray(get_patches(l, lc, fc_shape))
                for l, lc in zip(labels, centers)]
        y_fc = np.concatenate(y_fc)
        y = y.astype(dtype=np.bool)
        y_fc = y_fc.astype(dtype=np.bool)
        y = [
            keras.utils.to_categorical(y, num_classes=5),
            keras.utils.to_categorical(y_fc, num_classes=5).reshape((len(y_fc), -1, 5))
        ]

    return x, y


def transfer_learning(
        net_domain,
        net,
        data,
        x,
        y,
        train_roi
):
    c = color_codes()
    options = parse_inputs()
    # Network hyperparameters
    epochs = options['epochs']
    net_epochs = options['net_epochs']
    batch_size = options['batch_size']

    # We prepare the layers for transfer learning
    net_domain_conv_layers = [l for l in net_domain.layers if 'conv' in l.name]
    net_conv_layers = sorted(
        [l for l in net.layers if 'conv' in l.name],
        cmp=lambda l1, l2: int(l1.name[7:]) - int(l2.name[7:])
    )

    # We freeze the convolutional layers for the final net
    for layer in net.layers:
        if not isinstance(layer, Dense):
            layer.trainable = False
    net_domain_params = np.sum([K.count_params(p) for p in set(net_domain.trainable_weights)])

    # We start retraining.
    # First we retrain the convolutional so the tumor rois appear similar after convolution, and then we
    # retrain the classifier with the new convolutional weights.
    print(c['c'] + '[' + strftime("%H:%M:%S") + ']    ' + c['g'] + 'Training the models ' + c['nc'] +
          c['b'] + '(%d patches)' % len(x) + c['nc'])
    conv_data = net_domain.predict(np.expand_dims(train_roi, axis=0), batch_size=1)
    for e in range(epochs):
        print(c['b'] + 'Epoch %d/%d ' % (e+1, epochs) + c['nc'])
        print(''.join([' ']*14) + c['g'] + c['b'] + 'Domain' + c['nc'] + c['g'] + ' net ' + c['nc'] +
              c['b'] + '(%d parameters)' % net_domain_params + c['nc'])
        net_domain.fit(np.expand_dims(data, axis=0), conv_data, epochs=1, batch_size=1)
        for layer in net.layers:
            if isinstance(layer, Dense):
                if layer.name in ['core', 'tumor', 'enhancing']:
                    layer.trainable = False
                else:
                    layer.trainable = True
        net.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
        net_params = np.sum([K.count_params(p) for p in set(net.trainable_weights)])
        print(''.join([' ']*14) + c['g'] + c['b'] + 'Original (dense)' + c['nc'] + c['g'] + ' net ' + c['nc'] +
              c['b'] + '(%d parameters)' % net_params + c['nc'])
        net.fit(x, y, epochs=net_epochs, batch_size=batch_size)
        for layer in net.layers:
            if isinstance(layer, Dense):
                if layer.name in ['core', 'tumor', 'enhancing']:
                    layer.trainable = True
                else:
                    layer.trainable = False
        net.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
        net_params = np.sum([K.count_params(p) for p in set(net.trainable_weights)])
        print(''.join([' ']*14) + c['g'] + c['b'] + 'Original (out)' + c['nc'] + c['g'] + ' net ' + c['nc'] +
              c['b'] + '(%d parameters)' % net_params + c['nc'])
        net.fit(x, y, epochs=net_epochs, batch_size=batch_size)
    # We transfer the convolutional weights after retraining the net
    for l_new, l_orig in zip(net_domain_conv_layers, net_conv_layers):
        l_orig.set_weights(l_new.get_weights())


def test_network(net, p, batch_size, patch_size, sufix='', centers=None, filename=None, fcn=False):

    c = color_codes()
    p_name = p[0].rsplit('/')[-2]
    patient_path = '/'.join(p[0].rsplit('/')[:-1])
    outputname = filename if filename is not None else 'deep-brats17.test.' + sufix
    outputname_path = os.path.join(patient_path, outputname + '.nii.gz')
    roiname = os.path.join(patient_path, outputname + '.roi.nii.gz')
    try:
        image = load_nii(outputname_path).get_data()
        load_nii(roiname)
    except IOError:
        print(c['c'] + '[' + strftime("%H:%M:%S") + ']    ' + c['g'] + 'Testing ' +
              c['b'] + sufix + c['nc'] + c['g'] + ' network' + c['nc'])
        roi_nii = load_nii(p[0])
        roi = roi_nii.get_data().astype(dtype=np.bool)
        centers = get_mask_voxels(roi) if centers is None else centers
        test_samples = np.count_nonzero(roi)
        image = np.zeros_like(roi).astype(dtype=np.uint8)
        print(c['c'] + '[' + strftime("%H:%M:%S") + ']    ' + c['g'] +
              '<Creating the probability map ' + c['b'] + p_name + c['nc'] + c['g'] +
              ' (%d samples)>' % test_samples + c['nc'])

        n_centers = len(centers)
        image_list = [load_norm_list(p)]
        is_roi = False
        for i in range(0, n_centers, 2097152):
            print('%f%% tested (step %d)' % (100.0 * i / n_centers, (i / batch_size) + 1), end='\r')
            sys.stdout.flush()
            centers_i = [centers[i:i + batch_size]]
            x = get_patches_list(image_list, centers_i, patch_size, True)
            x = np.concatenate(x).astype(dtype=np.float32)
            y_pr_pred = net.predict(x)

            [x, y, z] = np.stack(centers_i[0], axis=1)

            if fcn:
                y_pr_pred = y_pr_pred[0]
                tumor = np.argmax(y_pr_pred, axis=1).astype(np.bool)
                is_roi = False
            else:
                if isinstance(y_pr_pred, list):
                    tumor = np.argmax(y_pr_pred[0], axis=1)
                    y_pr_pred = y_pr_pred[-1]
                    is_roi = False
                else:
                    tumor = np.argmax(y_pr_pred, axis=1)
                    is_roi = True

            # We store the ROI
            roi = np.zeros_like(roi).astype(dtype=np.uint8)
            roi[x, y, z] = tumor
            # We store the results
            y_pred = np.argmax(y_pr_pred, axis=1)
            image[x, y, z] = tumor if is_roi else y_pred

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


def create_new_network(patch_size, filters_list, kernel_size_list):
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


def get_domain_image(
        p,
        net_name,
        net_new_name,
        net_roi_name,
        image_o,
        patch_size,
        outputname,
        fcn=False,
        roi=None
):
    options = parse_inputs()
    c = color_codes()

    path = options['dir_name']
    train_data, train_labels = get_names_from_path(os.path.join(path, '../Brats17Test-Training'), options)

    # Prepare the net hyperparameters
    batch_size = options['batch_size']
    conv_blocks = options['conv_blocks']
    n_filters = options['n_filters']
    filters_list = n_filters if len(n_filters) > 1 else n_filters * conv_blocks
    conv_width = options['conv_width']
    kernel_size_list = conv_width if isinstance(conv_width, list) else [conv_width] * conv_blocks
    options_s = 'e%d.E%d.D%d.' % (options['epochs'], options['net_epochs'], options['down_factor'])

    p_name = p[0].rsplit('/')[-2]
    patient_path = '/'.join(p[0].rsplit('/')[:-1])

    try:
        image_d = load_nii(os.path.join(patient_path, outputname + '.nii.gz')).get_data()
    except IOError:
        net_roi = keras.models.load_model(net_roi_name)
        net_orig = keras.models.load_model(net_name)
        net_orig_conv_layers = sorted(
            [l for l in net_orig.layers if 'conv' in l.name],
            cmp=lambda x, y: int(x.name[7:]) - int(y.name[7:])
        )

        # Now let's create the domain network and train it
        try:
            net_new = keras.models.load_model(net_new_name)
            net_new_conv_layers = [l for l in net_new.layers if 'conv' in l.name]
            # We transfer the convolutional weights after retraining the net
            for l_new, l_orig in zip(net_new_conv_layers, net_orig_conv_layers):
                l_orig.set_weights(l_new.get_weights())
        except IOError:
            # First we get the tumor ROI
            if roi is None:
                image_r = test_network(net_roi, p, batch_size, patch_size, filename=outputname + '.tumor', fcn=fcn)
                roi = np.logical_and(image_r.astype(dtype=np.bool), image_o.astype(dtype=np.bool))
                if np.count_nonzero(roi) == 0:
                    roi = image_r
            p_images = np.stack(load_norm_list(p)).astype(dtype=np.float32)
            data, clip = clip_to_roi(p_images, roi)
            data_s = c['g'] + c['b'] + 'x'.join(['%d' % i_len for i_len in data.shape[1:]]) + c['nc']
            print(c['c'] + '[' + strftime("%H:%M:%S") + ']    ' + c['g'] + 'Preparing ' + c['b'] + 'domain' + c['nc'] +
                  c['g'] + ' data ' + c['nc'] + c['g'] + '(shape = ' + data_s + c['g'] + ')' + c['nc'])
            # We prepare the zoomed tumors for training
            roi_name = os.path.join(path, p_name + outputname + '.roi.pkl')
            mask_name = os.path.join(path, p_name + outputname + '.mask.pkl')
            image_name = os.path.join(path, p_name + outputname + '.image.pkl')
            rate_name = os.path.join(path, p_name + outputname + '.rate.pkl')
            try:
                train_roi = pickle.load(open(roi_name, 'rb'))
                train_mask = pickle.load(open(mask_name, 'rb'))
                train_image = pickle.load(open(image_name, 'rb'))
                train_rate = pickle.load(open(rate_name, 'rb'))
            except IOError:
                train_num, train_roi, train_rate = get_best_roi(data, train_data, train_labels)
                train_image = np.stack(load_norm_list(train_data[train_num])).astype(dtype=np.float32)
                train_mask = load_nii(train_labels[train_num]).get_data().astype(dtype=np.uint8)
                pickle.dump(train_roi, open(roi_name, 'wb'))
                pickle.dump(train_mask, open(mask_name, 'wb'))
                pickle.dump(train_image, open(image_name, 'wb'))
                pickle.dump(train_rate, open(rate_name, 'wb'))
            _, train_clip = clip_to_roi(train_image, train_mask)

            train_x = zoom(train_image, train_rate)
            train_y = zoom(train_mask, train_rate[1:], order=0)

            # We create the domain network
            net_new = create_new_network(data.shape[1:], filters_list, kernel_size_list) if not fcn\
                else get_brats_net(data.shape[1:], filters_list, kernel_size_list, [32]*4, 5, domain=True)
            net_new_conv_layers = [l for l in net_new.layers if 'conv' in l.name]
            for l_new, l_orig in zip(net_new_conv_layers, net_orig_conv_layers):
                l_new.set_weights(l_orig.get_weights())

            # Transfer learning
            train_centers_r = [range(int(cl[0] * tr), int(cl[1] * tr)) for cl, tr in zip(train_clip, train_rate[1:])]
            train_centers = list(product(*train_centers_r))

            print(c['c'] + '[' + strftime("%H:%M:%S") + ']    ' + c['g'] + 'Preparing ' + c['b'] + 'domain' + c['nc'] +
                  c['g'] + ' patches ' + ['b'] + '(%d patches)' % len(train_centers) + c['nc'])
            train_x, train_y = data_loading(train_x, train_y, train_centers, fcn=fcn)
            transfer_learning(net_new, net_orig, data, train_x, train_y, train_roi)
            net_new.save(net_new_name)

        image_d = test_network(net_orig, p, batch_size, patch_size, sufix=options_s + 'domain', fcn=fcn)

    return image_d


def get_best_roi(base_roi, image_list, labels_list):
    best_rank = -np.inf
    best_name = None
    best_image = None
    best_roi = None
    best_rate = None
    for i, (p, gt_name) in enumerate(zip(image_list, labels_list)):
        im = np.stack(load_norm_list(p)).astype(dtype=np.float32)
        gt_nii = load_nii(gt_name)
        gt = gt_nii.get_data().astype(dtype=np.bool)
        im_clipped, clip = clip_to_roi(im, gt)
        zoom_rate = [float(b_len)/i_len for b_len, i_len in zip(base_roi.shape[1:], im_clipped.shape[1:])]
        im_roi = zoom(im_clipped, zoom=[1.0] + zoom_rate)
        nu_rank = ssim(np.moveaxis(base_roi, 0, -1), np.moveaxis(im_roi, 0, -1), multichannel=True)
        if nu_rank > best_rank:
            best_rank = nu_rank
            best_roi = im_roi
            best_image = i
            best_rate = [1.0] + zoom_rate
            best_name = p
        print(''.join([' ']*14) + 'Image %s - SSIM = %f' % (p[0].rsplit('/')[-2], nu_rank))
    print(''.join([' '] * 14) + 'Best rank = %s - SSIM = %f' % (best_name[0].rsplit('/')[-2], best_rank))
    return best_image, best_roi, best_rate


def main():
    options = parse_inputs()
    c = color_codes()

    path = options['dir_name']
    test_data, test_labels = get_names_from_path(path, options)
    net_name = os.path.join(path, 'baseline-brats2017.D50.f.p13.c3c3c3c3c3.n32n32n32n32n32.d256.e50.mdl')
    net_name2 = os.path.join(path, 'brats2017-seg.D100.p17.c3c3c3c3.n32n32n32n32.d256.e5.E10.e9.best.hdf5')

    # Prepare the net hyperparameters
    patch_width = options['patch_width']
    patch_size = (patch_width, patch_width, patch_width)
    batch_size = options['batch_size']
    options_s = 'e%d.E%d.D%d.' % (options['epochs'], options['net_epochs'], options['down_factor'])

    print(c['c'] + '[' + strftime("%H:%M:%S") + '] ' + 'Starting testing' + c['nc'])
    # Testing. We retrain the convolutionals and then apply testing. We also check the results without doing it.
    dsc_results_o = list()
    dsc_results_o2 = list()
    dsc_results_d = list()
    dsc_results_d2 = list()

    for i, (p, gt_name) in enumerate(zip(test_data, test_labels)):
        p_name = p[0].rsplit('/')[-2]
        patient_path = '/'.join(p[0].rsplit('/')[:-1])
        print(c['c'] + '[' + strftime("%H:%M:%S") + ']  ' + c['nc'] + 'Case ' + c['c'] + c['b'] + p_name + c['nc'] +
              c['c'] + ' (%d/%d):' % (i + 1, len(test_data)) + c['nc'])
        try:
            image_o = load_nii(os.path.join(patient_path, p_name + '.nii.gz')).get_data()
        except IOError:
            # First let's test the original network
            net_orig = keras.models.load_model(net_name)
            image_o = test_network(net_orig, p, batch_size, patch_size, sufix='original', filename=p_name)

        # First let's test the original network
        net_orig2 = keras.models.load_model(net_name2)
        image_o2 = test_network(net_orig2, p, batch_size, (17, 17, 17), sufix='original.2', fcn=True)

        outputname = 'deep-brats17.test.' + options_s + 'domain'
        net_new_name = os.path.join(path, 'domain-exp-brats2017.' + options_s + p_name + '.mdl')
        net_roi_name = os.path.join(path, 'CBICA-brats2017.D25.p13.c3c3c3c3c3.n32n32n32n32n32.d256.e50.mdl')
        image_d = get_domain_image(
            p,
            net_name,
            net_new_name,
            net_roi_name,
            image_o,
            patch_size,
            outputname
        )

        net_roi_name2 = os.path.join(path, 'roi.2-brats2017.' + options_s + p_name + '.mdl')
        get_domain_image(
            p,
            net_name2,
            net_roi_name2,
            net_name2,
            image_o2,
            (17, 17, 17),
            'dummy',
            roi=load_nii(p[0]).get_data().astype(np.bool),
            fcn=True
        )
        outputname = 'deep-brats17.test.' + options_s + 'domain.2'
        net_new_name2 = os.path.join(path, 'domain-exp.2-brats2017.' + options_s + p_name + '.mdl')
        image_d2 = get_domain_image(
            p,
            net_name2,
            net_new_name2,
            net_roi_name2,
            image_o2,
            (17, 17, 17),
            outputname,
            fcn=True
        )

        if options['use_dsc']:
            results_o = check_dsc(gt_name, image_o)
            dsc_results_o.append(results_o)
            results_o2 = check_dsc(gt_name, image_o2)
            dsc_results_o2.append(results_o2)
            results_d = check_dsc(gt_name, image_d)
            dsc_results_d.append(results_d)
            results_d2 = check_dsc(gt_name, image_d2)
            dsc_results_d2.append(results_d2)

            subject_name = c['c'] + c['b'] + '%s' + c['nc']
            dsc_string = c['g'] + '/'.join(['%f']*len(results_o)) + c['nc']
            text = subject_name + ' DSC: ' + dsc_string
            results = (p_name,) + tuple(results_o)
            print(''.join([' ']*14) + 'Original   ' + text % results)
            results = (p_name,) + tuple(results_o2)
            print(''.join([' '] * 14) + 'FCN        ' + text % results)
            results = (p_name,) + tuple(results_d)
            print(''.join([' ']*14) + 'Domain     ' + text % results)
            results = (p_name,) + tuple(results_d2)
            print(''.join([' ']*14) + 'Domain FCN ' + text % results)

    if options['use_dsc']:
        f_dsc_o = tuple([np.array([dsc[i] for dsc in dsc_results_o if len(dsc) > i]).mean() for i in range(3)])
        f_dsc_o2 = tuple([np.array([dsc[i] for dsc in dsc_results_o2 if len(dsc) > i]).mean() for i in range(3)])
        f_dsc_d = tuple([np.array([dsc[i] for dsc in dsc_results_d if len(dsc) > i]).mean() for i in range(3)])
        f_dsc_d2 = tuple([np.array([dsc[i] for dsc in dsc_results_d2 if len(dsc) > i]).mean() for i in range(3)])
        f_dsc = f_dsc_o + f_dsc_o2 + f_dsc_d + f_dsc_d2
        print('Final results DSC: (%f/%f/%f) - (%f/%f/%f) vs (%f/%f/%f) - (%f/%f/%f)' % f_dsc)


if __name__ == '__main__':
    main()
