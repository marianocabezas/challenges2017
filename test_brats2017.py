from __future__ import print_function
import argparse
import os
import sys
from itertools import product
from time import strftime
import numpy as np
import keras
from keras.models import Model
from keras.layers import Conv3D, Dropout, Input, Reshape, Lambda
from nibabel import load as load_nii
from utils import color_codes, get_biggest_region
from data_creation import load_norm_list
from data_creation import load_patch_batch_generator_test, load_patch_batch_generator_train
from data_manipulation.generate_features import get_mask_voxels
from data_manipulation.metrics import dsc_seg
from scipy.ndimage.interpolation import zoom
from skimage.measure import compare_ssim as ssim


def parse_inputs():
    # I decided to separate this function, for easier acces to the command line parameters
    parser = argparse.ArgumentParser(description='Test different nets with 3D data.')
    parser.add_argument('-f', '--folder', dest='dir_name', default='/home/mariano/DATA/Brats17Test/')
    parser.add_argument('-i', '--patch-width', dest='patch_width', type=int, default=13)
    parser.add_argument('-k', '--kernel-size', dest='conv_width', nargs='+', type=int, default=3)
    parser.add_argument('-c', '--conv-blocks', dest='conv_blocks', type=int, default=5)
    parser.add_argument('-b', '--batch-size', dest='batch_size', type=int, default=2048)
    parser.add_argument('-D', '--down-factor', dest='dfactor', type=int, default=50)
    parser.add_argument('-n', '--num-filters', action='store', dest='n_filters', nargs='+', type=int, default=[32])
    parser.add_argument('-N', '--num-images', action='store', dest='n_images', type=int, default=25)
    parser.add_argument('-e', '--epochs', action='store', dest='epochs', type=int, default=25)
    parser.add_argument('-q', '--queue', action='store', dest='queue', type=int, default=100)
    parser.add_argument('--no-flair', action='store_false', dest='use_flair', default=True)
    parser.add_argument('--no-t1', action='store_false', dest='use_t1', default=True)
    parser.add_argument('--no-t1ce', action='store_false', dest='use_t1ce', default=True)
    parser.add_argument('--no-t2', action='store_false', dest='use_t2', default=True)
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


def transfer_learning(net_domain, net, data, train_image, train_roi, train_labels, train_centers, options):
    epochs = options['epochs']
    patch_width = options['patch_width']
    patch_size = (patch_width, patch_width, patch_width)
    batch_size = options['batch_size']
    # Data loading parameters
    queue = options['queue']
    # We prepare the layers for transfer learning
    net_domain_conv_layers = [l for l in net_domain.layers if 'conv' in l.name]
    net_conv_layers = sorted(
        [l for l in net.layers if 'conv' in l.name],
        lambda x, y: int(x.name[7:]) - int(y.name[7:])
    )
    # We freeze the convolutional layers for the final net
    for _ in range(epochs):
        conv_data = net_domain.predict(np.expand_dims(train_roi, axis=0), batch_size=1)
        net_domain.fit(np.expand_dims(data, axis=0), conv_data, epochs=1, batch_size=1)
        for l_new, l_orig in zip(net_domain_conv_layers, net_conv_layers):
            l_orig.set_weights(l_new.get_weights())
        train_steps_per_epoch = -(-np.size(train_roi) / batch_size)
        net.fit_generator(
            generator=load_patch_batch_generator_train(
                image_list=[train_image],
                label_names=train_labels,
                centers=train_centers,
                batch_size=batch_size,
                size=patch_size,
                nlabels=4,
                dfactor=1,
                preload=True,
                split=True
            ),
            steps_per_epoch=train_steps_per_epoch,
            max_q_size=queue,
            epochs=epochs
        )


def test_network(net, p, batch_size, patch_size, queue, sufix='', centers=None):
    c = color_codes()
    p_name = p[0].rsplit('/')[-2]
    patient_path = '/'.join(p[0].rsplit('/')[:-1])
    outputname = os.path.join(patient_path, 'deep-brats17.test.' + sufix + '.nii.gz')
    roiname = os.path.join(patient_path, 'deep-brats17.orig.test.' + sufix + '.roi.nii.gz')
    try:
        image = load_nii(outputname).get_data()
        load_nii(roiname)
    except IOError:
        print(c['c'] + '[' + strftime("%H:%M:%S") + ']    ' + c['g'] + 'Testing ' +
              c['b'] + sufix + c['nc'] + c['g'] + ' network' + c['nc'])
        roi_nii = load_nii(p[0])
        roi = roi_nii.get_data().astype(dtype=np.bool)
        centers = get_mask_voxels(roi) if centers is not None else centers
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
                preload=True,
            ),
            steps=test_steps_per_epoch,
            max_q_size=queue
        )
        print(' '.join(['']*50), end='\r')
        sys.stdout.flush()
        [x, y, z] = np.stack(centers, axis=1)

        tumor = np.argmax(y_pr_pred[0], axis=1)
        y_pr_pred = y_pr_pred[-1]
        roi = np.zeros_like(roi).astype(dtype=np.uint8)
        roi[x, y, z] = tumor
        roi_nii.get_data()[:] = roi
        roi_nii.to_filename(roiname)

        y_pred = np.argmax(y_pr_pred, axis=1)

        image[x, y, z] = y_pred
        # Post-processing (Basically keep the biggest connected region)
        image = get_biggest_region(image)
        print(c['g'] + '                   -- Saving image ' + c['b'] + outputname + c['nc'])
        roi_nii.get_data()[:] = image
        roi_nii.to_filename(outputname)
    return image, p_name


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


def get_best_image(base_image, image_list, base_shape):
    best_rank = 0
    best_image = None
    for p in image_list:
        im = np.array(load_norm_list(p), dtype=np.float32)
        shape_diff = map(lambda (x, y): x - y, zip(im.shape, base_shape))
        im = np.pad(im, tuple(map(lambda x: (x/2, x-x/2), shape_diff)), 'constant')
        nu_rank = ssim(base_image.flatten(), im.flatten())
        if nu_rank > best_rank:
            best_rank = nu_rank
            best_image = im
        print(''.join([' ']*14) + 'Image %s - SSIM = %f' % (p[0].rsplit('/')[-2], nu_rank))
    return best_image


def get_best_roi(base_roi, image_list, labels_list):
    best_rank = 0
    best_roi = None
    best_image = None
    best_labels = None
    for p, gt_name in zip(image_list, labels_list):
        im = np.array(load_norm_list(p), dtype=np.float32)
        gt_nii = load_nii(gt_name)
        gt = gt_nii.get_data().astype(dtype=np.bool)
        zoom_rate = [float(b_len)/i_len for b_len, i_len in zip(base_roi.shape[1:], im.shape[1:])]
        im_roi = zoom(gt, zoom=[1.0] + zoom_rate)
        nu_rank = ssim(base_roi.flatten(), im.flatten())
        if nu_rank > best_rank:
            best_rank = nu_rank
            best_roi = im_roi
            best_image = im
            best_labels = gt_name
        print(''.join([' ']*14) + 'Image %s - SSIM = %f' % (p[0].rsplit('/')[-2], nu_rank))
    return best_image, best_roi, best_labels


def clip_to_roi(images, roi):
    # We clip with padding for patch extraction
    min_coord = np.stack(np.nonzero(roi.astype(dtype=np.bool))).min(axis=1)
    max_coord = np.stack(np.nonzero(roi.astype(dtype=np.bool))).max(axis=1)

    clip = np.array([(min_c, max_c) for min_c, max_c in zip(min_coord, max_coord)], dtype=np.uint8)
    im_clipped = images[:, clip[0, 0]:clip[0, 1], clip[1, 0]:clip[1, 1], clip[2, 0]:clip[2, 1]]

    return im_clipped, clip


def main():
    options = parse_inputs()
    c = color_codes()

    path = options['dir_name']
    test_data, test_labels = get_names_from_path(path, options)
    train_data, train_labels = get_names_from_path(os.path.join(path, '../Brats17Test-Training'), options)
    net_name = os.path.join(path, 'baseline-brats2017.D50.f.p13.c3c3c3c3c3.n32n32n32n32n32.d256.e50.mdl')

    # Prepare the net hyperparameters
    epochs = options['epochs']
    patch_width = options['patch_width']
    patch_size = (patch_width, patch_width, patch_width)
    batch_size = options['batch_size']
    conv_blocks = options['conv_blocks']
    n_filters = options['n_filters']
    filters_list = n_filters if len(n_filters) > 1 else n_filters*conv_blocks
    conv_width = options['conv_width']
    kernel_size_list = conv_width if isinstance(conv_width, list) else [conv_width]*conv_blocks
    # Data loading parameters
    queue = options['queue']

    print(c['c'] + '[' + strftime("%H:%M:%S") + '] ' + 'Starting testing' + c['nc'])
    # Testing. We retrain the convolutionals and then apply testing. We also check the results without doing it.
    dsc_results = list()

    net_roi_name = os.path.join(path, 'CBICA-brats2017.D25.p13.c3c3c3c3c3.n32n32n32n32n32.d256.e50.mdl')
    net_roi = keras.models.load_model(net_roi_name)
    for i, (p, gt_name) in enumerate(zip(test_data, test_labels)):
        print(c['c'] + '[' + strftime("%H:%M:%S") + ']  ' + c['nc'] +
              'Case ' + c['c'] + c['b'] + '%d/%d: ' % (i + 1, len(test_data)) + c['nc'])

        # First let's test the original network
        net_orig = keras.models.load_model(net_name)
        net_orig_conv_layers = sorted(
            [l for l in net_orig.layers if 'conv' in l.name],
            lambda x, y: int(x.name[7:]) - int(y.name[7:])
        )
        gt_nii = load_nii(gt_name)
        gt = np.copy(gt_nii.get_data()).astype(dtype=np.uint8)
        labels = np.unique(gt.flatten())

        image_o, p_name = test_network(net_orig, p, batch_size, patch_size, queue, sufix='original')

        results_o = [dsc_seg(gt == l, image_o == l) for l in labels[1:]]
        subject_name = c['c'] + c['b'] + '%s' + c['nc']
        dsc_string = c['g'] + '/'.join(['%f']*len(results_o)) + c['nc']
        text = subject_name + ' DSC: ' + dsc_string

        # Now let's create the domain network and train it
        net_new_name = os.path.join(path, 'domain-brats2017.' + p_name + '.mdl')
        try:
            net_new = keras.models.load_model(net_new_name)
            net_new_conv_layers = [l for l in net_new.layers if 'conv' in l.name]
        except IOError:
            print(c['c'] + '[' + strftime("%H:%M:%S") + ']    ' + c['g'] + 'Preparing ' +
                  c['b'] + 'domain' + c['nc'] + c['g'] + ' net' + c['nc'])
            # First we get the tumor ROI
            # image_r, p_name = test_network(net_roi, p, batch_size, patch_size, queue, sufix='roi')
            # p_images = np.array(load_norm_list(p), dtype=np.float32)
            # data, clip = clip_to_roi(p_images, image_r, patch_size)
            # train_image, train_roi, train_labels = get_best_roi(data, train_data, train_labels)

            # We get the base image and the training image downscaled (maybe I could break them in "big patches")
            image = np.array(load_norm_list(p), dtype=np.float32)
            train_image = zoom(get_best_image(image, train_data, image.shape), [1, 0.5, 0.5, 0.5])
            data = zoom(image, [1, 0.5, 0.5, 0.5])
            # We create the domain network
            net_new = create_new_network(data.shape[1:], filters_list, kernel_size_list)
            net_new_conv_layers = [l for l in net_new.layers if 'conv' in l.name]
            for l_new, l_orig in zip(net_new_conv_layers, net_orig_conv_layers):
                l_new.set_weights(l_orig.get_weights())
            # Getting the "labeled data"
            print(c['c'] + '[' + strftime("%H:%M:%S") + ']    ' + c['g'] + 'Preparing ' +
                  c['b'] + 'training data' + c['nc'])
            conv_data = net_new.predict(np.expand_dims(train_image, axis=0), batch_size=1)
            # Training part
            print(c['c'] + '[' + strftime("%H:%M:%S") + ']    ' +
                  c['g'] + 'Training the model with %d images ' % len(conv_data) +
                  c['b'] + '(%d parameters)' % net_new.count_params() + c['nc'])
            print(net_new.summary())
            # Transfer learning
            # train_centers = list(product([range(c[0], c[1]) for c in clip]))
            # transfer_learning(net_new, net_orig, data, train_image, train_roi, train_centers, options)
            net_new.fit(np.expand_dims(data, axis=0), conv_data, epochs=epochs, batch_size=1)
            net_new.save(net_new_name)

        # Now we transfer the new weights an re-test
        for l_new, l_orig in zip(net_new_conv_layers, net_orig_conv_layers):
            l_orig.set_weights(l_new.get_weights())

        image_d, p_name = test_network(net_orig, p, batch_size, patch_size, queue, sufix='domain')

        results_d = [dsc_seg(gt == l, image_d == l) for l in labels[1:]]
        results = (p_name,) + tuple(results_o)
        print(''.join([' ']*14) + 'Original ' + text % results)
        results = (p_name,) + tuple(results_d)
        print(''.join([' ']*14) + 'Domain   ' + text % results)

        dsc_results.append(results_o + results_d)

    f_dsc = tuple(np.asarray(dsc_results).mean())
    print('Final results DSC: ' + '/'.join(['%f']*len(f_dsc)) % f_dsc)


if __name__ == '__main__':
    main()
