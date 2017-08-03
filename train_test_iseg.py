from __future__ import print_function
import argparse
import os
from time import strftime
import numpy as np
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from nibabel import load as load_nii
from nibabel import save as save_nii
from utils import color_codes, nfold_cross_validation, get_patient_info
from itertools import izip
from data_creation import load_patches_train, get_cnn_centers
from data_creation import load_patch_batch_generator_test
from data_manipulation.generate_features import get_mask_voxels
from data_manipulation.metrics import dsc_seg
from nets import get_iseg_baseline, get_iseg_experimental1, get_iseg_experimental2, get_iseg_experimental3
from nets import get_iseg_experimental4


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
    parser.add_argument('-p', '--patience', dest='patience', type=int, default=5)
    parser.add_argument('-n', '--num-filters', action='store', dest='n_filters', nargs='+', type=int, default=[32])
    parser.add_argument('-e', '--epochs', action='store', dest='epochs', type=int, default=50)
    parser.add_argument('-E', '--experimental', action='store', dest='experimental', type=int, default=0)
    parser.add_argument('-q', '--queue', action='store', dest='queue', type=int, default=100)
    parser.add_argument('-s', '--sequential', action='store_true', dest='sequential', default=False)
    parser.add_argument('--preload', action='store_true', dest='preload', default=False)
    parser.add_argument('--t1', action='store', dest='t1', default='-T1.hdr')
    parser.add_argument('--t2', action='store', dest='t2', default='-T2.hdr')
    parser.add_argument('--labels', action='store', dest='labels', default='-label.hdr')
    return vars(parser.parse_args())


def list_directories(path):
    return filter(os.path.isdir, [os.path.join(path, f) for f in os.listdir(path)])


def get_sufix(options):
    # Prepare the net architecture parameters
    dfactor = options['dfactor']
    # Prepare the net hyperparameters
    epochs = options['epochs']
    patch_width = options['patch_width']
    dense_size = options['dense_size']
    conv_blocks = options['conv_blocks']
    n_filters = options['n_filters']
    filters_list = n_filters if len(n_filters) > 1 else n_filters * conv_blocks
    conv_width = options['conv_width']
    kernel_size_list = conv_width if isinstance(conv_width, list) else [conv_width] * conv_blocks
    experimental = options['experimental']

    # Prepare the sufix that will be added to the results for the net and images
    filters_s = 'n'.join(['%d' % nf for nf in filters_list])
    conv_s = 'c'.join(['%d' % cs for cs in kernel_size_list])
    exp_s = 'experimental-%d' % experimental if experimental else 'baseline'
    params_s = (exp_s, dfactor, patch_width, conv_s, filters_s, dense_size, epochs)

    return '.%s.D%d.p%d.c%s.n%s.d%d.e%d.' % params_s


def get_names_from_path(options):
    path = options['dir_name']

    patients = range(10)

    # Prepare the names
    t1_names = [os.path.join(path, 'subject-%d' % (p+1) + options['t1']) for p in patients]
    t2_names = [os.path.join(path, 'subject-%d' % (p+1) + options['t2']) for p in patients]
    label_names = np.array([os.path.join(path, 'subject-%d' % (p+1) + options['labels']) for p in patients])
    image_names = np.stack([t1_names, t2_names], axis=1)

    return image_names, label_names


def train_net(fold_n, train_data, train_labels, options):
    # Prepare the net architecture parameters
    dfactor = options['dfactor']
    # Prepare the net hyperparameters
    epochs = options['epochs']
    patch_width = options['patch_width']
    patch_size = (patch_width,) * 3
    batch_size = options['batch_size']
    dense_size = options['dense_size']
    conv_blocks = options['conv_blocks']
    nfilters = options['n_filters']
    filters_list = nfilters if len(nfilters) > 1 else nfilters*conv_blocks
    conv_width = options['conv_width']
    kernel_size_list = conv_width if isinstance(conv_width, list) else [conv_width]*conv_blocks
    experimental = options['experimental']

    fc_width = patch_width - sum(kernel_size_list) + conv_blocks
    fc_shape = (fc_width,) * 3
    # Data loading parameters
    preload = options['preload']

    # Prepare the sufix that will be added to the results for the net and images
    path = options['dir_name']
    sufix = get_sufix(options)

    net_name = os.path.join(path, 'iseg2017.fold%d' % fold_n + sufix + 'mdl')

    c = color_codes()
    try:
        net = load_model(net_name)
    except IOError:
        # Data loading
        train_centers = get_cnn_centers(train_data[:, 0], train_labels)
        train_samples = len(train_centers) / dfactor
        print(c['c'] + '[' + strftime("%H:%M:%S") + ']    ' + c['g'] + 'Loading data ' +
              c['b'] + '(%d centers)' % len(train_centers) + c['nc'])
        x, y = load_patches_train(
            image_names=train_data,
            label_names=train_labels,
            centers=train_centers,
            size=patch_size,
            fc_shape=fc_shape,
            nlabels=4,
            dfactor=dfactor,
            preload=preload,
            split=True,
            iseg=True,
            experimental=experimental,
            datatype=np.float32
        )
        # NET definition using Keras
        print(c['c'] + '[' + strftime("%H:%M:%S") + ']    ' + c['g'] + 'Creating and compiling the model ' +
              c['b'] + '(%d samples)' % train_samples + c['nc'])
        input_shape = (2,) + patch_size
        # This architecture is based on the functional Keras API to introduce 3 output paths:
        # - Whole tumor segmentation
        # - Core segmentation (including whole tumor)
        # - Whole segmentation (tumor, core and enhancing parts)
        # The idea is to let the network work on the three parts to improve the multiclass segmentation.
        network_func = [
            get_iseg_baseline,
            get_iseg_experimental1,
            get_iseg_experimental2,
            get_iseg_experimental3,
            get_iseg_experimental4
        ]
        net = network_func[experimental](
            input_shape,
            filters_list,
            kernel_size_list,
            dense_size
        )

        print(c['c'] + '[' + strftime("%H:%M:%S") + ']    ' +
              c['g'] + 'Training the model ' + c['b'] + '(%d parameters)' % net.count_params() + c['nc'])
        print(net.summary())
        checkpooint =  'iseg2017.fold%d' % fold_n + sufix + '{epoch:02d}.{brain_val_acc:.2f}.hdf5'
        callbacks = [
            EarlyStopping(monitor='val_brain_loss', patience=options['patience']),
            ModelCheckpoint(os.path.join(path, checkpooint), monitor='val_brain_loss', save_best_only=True)
        ]
        net.fit(x, y, batch_size=batch_size, validation_split=0.25, epochs=epochs, callbacks=callbacks)
        net.save(net_name)
        # net.load_weights(os.path.join(path, checkpooint))
    return net


def check_image_list(patients_list, options):
    sufix = get_sufix(options)
    try:
        for p in patients_list:
            p_name, patient_path = get_patient_info(p)
            outputname = os.path.join(patient_path, 'deep-' + p_name + sufix + 'brain.hdr')
            np.squeeze(load_nii(outputname))
        return True
    except IOError:
        return False


def test_net(net, p, gt_name, options):
    # Testing hyperparameters
    patch_width = options['patch_width']
    patch_size = (patch_width, patch_width, patch_width)
    batch_size = options['batch_size']
    # Data loading parameters
    preload = options['preload']
    queue = options['queue']

    sufix = get_sufix(options)

    c = color_codes()
    p_name = '-'.join(p[0].rsplit('/')[-1].rsplit('.')[0].rsplit('-')[:-1])
    patient_path = '/'.join(p[0].rsplit('/')[:-1])
    outputname = os.path.join(patient_path, 'deep-' + p_name + sufix + 'brain.hdr')
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

        if options['experimental'] >= 3:
            y_pr_pred = y_pr_pred[:-1]

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
            roiname = os.path.join(patient_path, 'deep-' + p_name + im + 'roi.hdr')
            print(c['g'] + '                   -- Saving image ' + c['b'] + roiname + c['nc'])
            save_nii(gt_nii, roiname)

        y_pred = np.argmax(y_pr_pred[-1], axis=1)

        image[x, y, z] = y_pred
        gt_nii.get_data()[:] = np.expand_dims(image, axis=3)
        save_nii(gt_nii, outputname)

    return image, gt


def main():
    options = parse_inputs()
    c = color_codes()

    experimental = options['experimental']

    exp_s = c['b'] + '(experimental %d)' % experimental if experimental else c['b'] + '(baseline)'
    print(c['c'] + '[' + strftime("%H:%M:%S") + '] ' + 'Starting cross-validation ' + exp_s + c['nc'])
    # N-fold cross validation main loop (we'll do 2 training iterations with testing for each patient)
    data_names, label_names = get_names_from_path(options)
    folds = len(data_names)
    fold_generator = izip(nfold_cross_validation(data_names, label_names, n=folds), xrange(folds))
    dsc_results = list()
    for (train_data, train_labels, test_data, test_labels), i in fold_generator:
        print(c['c'] + '[' + strftime("%H:%M:%S") + ']  ' + c['nc'] + 'Fold %d/%d: ' % (i+1, folds) + c['g'] +
              'Number of training/testing images (%d=%d/%d)'
              % (len(train_data), len(train_labels), len(test_data)) + c['nc'])
        # Prepare the data relevant to the leave-one-out (subtract the patient from the dataset and set the path)
        # Also, prepare the network
        if not check_image_list(test_data, options):
            print(c['c'] + '[' + strftime("%H:%M:%S") + ']    ' + c['nc'] + c['g'] + 'Training' + c['nc'])
            net = train_net(i, train_data, train_labels, options)
        else:
            net = None

        # Then we test the net.
        print(c['c'] + '[' + strftime("%H:%M:%S") + ']    ' + c['nc'] + c['g'] + 'Testing' + c['nc'])
        for p, gt_name in zip(test_data, test_labels):
            image, gt = test_net(net, p, gt_name, options)
            p_name = '-'.join(p[0].rsplit('/')[-1].rsplit('.')[0].rsplit('-')[:-1])
            vals = np.unique(gt.flatten())

            gt_mask = np.sum(
                map(lambda (l, val): np.array(gt == val, dtype=np.uint8) * l, enumerate(vals)), axis=0
            )
            results = (
                str.capitalize(p_name),
                dsc_seg(gt_mask == 1, image == 1),
                dsc_seg(gt_mask == 2, image == 2),
                dsc_seg(gt_mask == 3, image == 3)
            )
            dsc_results.append(results)
            print('%s DSC: %f/%f/%f' % results)

    dsc_results = sorted(dsc_results, cmp=lambda x, y: int(x[0][8:]) - int(y[0][8:]))
    for results in dsc_results:
        print(c['c'] + '%s DSC: \033[32;1m%f/%f/%f' % results + c['nc'])
    f_dsc = tuple(np.asarray([results[1:] for results in dsc_results]).mean(axis=0))
    print(c['c'] + 'Final results DSC: \033[32;1m%f/%f/%f' % f_dsc + c['nc'])


if __name__ == '__main__':
    main()
