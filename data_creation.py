from __future__ import print_function
import sys
from operator import itemgetter, add
import numpy as np
from nibabel import load as load_nii
from data_manipulation.generate_features import get_mask_voxels, get_patches
from itertools import izip, chain, product
from scipy.ndimage.morphology import binary_dilation as imdilate
from numpy import logical_and as log_and
from numpy import logical_or as log_or
from numpy import logical_not as log_not
import keras


def norm(image):
    return (image - image[np.nonzero(image)].mean()) / image[np.nonzero(image)].std()


def subsample(center_list, sizes, random_state):
    np.random.seed(random_state)
    indices = [np.random.permutation(range(0, len(centers))).tolist()[:size]
               for centers, size in izip(center_list, sizes)]
    return [itemgetter(*idx)(centers) if idx else [] for centers, idx in izip(center_list, indices)]


def get_image_patches(image_list, centers, size, preload):
    print(len(image_list), centers, size)
    patches = [get_patches(image, centers, size) for image in image_list] if preload\
        else [get_patches(norm(load_nii(name).get_data()), centers, size) for name in image_list]
    return np.stack(patches, axis=1)


def get_patches_list(list_of_image_list, centers_list, size, preload):
    patch_list = [get_image_patches(image_list, centers, size, preload)
                  for image_list, centers in izip(list_of_image_list, centers_list) if centers]
    return patch_list


def centers_and_idx(centers, n_images):
    # This function is used to decompress the centers with image references into image indices and centers.
    centers = [list(map(lambda z: tuple(z[1]) if z[0] == im else [], centers)) for im in range(n_images)]
    idx = [map(lambda (a, b): [a] if b else [], enumerate(c)) for c in centers]
    centers = [filter(bool, c) for c in centers]
    idx = list(chain(*[chain(*i) for i in idx]))
    return centers, idx


def labels_generator(image_names):
    for patient in image_names:
        yield load_nii(patient).get_data()


def load_patch_batch_train(
        image_names,
        label_names,
        centers,
        batch_size,
        size,
        nlabels,
        dfactor=10,
        datatype=np.float32,
        preload=False,
        split=False,
        iseg=False
):
    image_list = [[norm(load_nii(image_name).get_data()) for image_name in patient]
                  for patient in image_names] if preload else image_names
    while True:
        gen = load_patch_batch_generator_train(
            image_list=image_list,
            label_names=label_names,
            center_list=centers,
            batch_size=batch_size,
            size=size,
            nlabels=nlabels,
            datatype=datatype,
            dfactor=dfactor,
            preload=preload,
            split=split,
            iseg=iseg
        )
        for x, y in gen:
            yield x, y


def load_patch_batch_generator_train(
        image_list,
        label_names,
        center_list,
        batch_size,
        size,
        nlabels,
        dfactor,
        preload=False,
        split=False,
        iseg=False,
        datatype=np.float32
):
    # The following line is important to understand the goal of the down scaling factor.
    # The idea of this parameter is to speed up training when using a large pool of samples, while trying
    # to retain the same variability. To accomplish that, at each epoch we shuffle the original samples (represented
    # by the center of the patch) and then get a subsample of this set. By randomly selecting at each step,
    # we can train with a larger dataset while also training each lesion with a smaller pool.
    # The random shuffle is important to guarantee the sample proportion in the original samples when the numbers
    # of epochs tends to infinite.
    batch_centers = np.random.permutation(center_list)[::dfactor]
    n_centers = len(batch_centers)
    n_images = len(image_list)
    for i in range(0, n_centers, batch_size):
        centers, idx = centers_and_idx(batch_centers[i:i + batch_size], n_images)
        x = get_patches_list(image_list, centers, size, preload)
        x = np.concatenate(filter(lambda z: z.any(), x)).astype(dtype=datatype)
        x[idx] = x
        y = [np.array([l[c] for c in lc]) for l, lc in izip(labels_generator(label_names), centers)]
        y = np.concatenate(y)
        y[idx] = y
        if split:
            if iseg:
                y = [keras.utils.to_categorical(y == (l+1), num_classes=2) for l in range(nlabels-1)] +\
                    [keras.utils.to_categorical(y, num_classes=nlabels)]
            else:
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
                        num_classes=nlabels
                    )
                ]
        else:
            y = keras.utils.to_categorical(y, num_classes=nlabels)
        yield (x, y)


def load_patch_batch_generator_test(
        image_names,
        centers,
        batch_size,
        size,
        preload=False,
        datatype=np.float32,
):
    while True:
        n_centers = len(centers)
        image_list = [norm(load_nii(image_name).get_data()) for image_name in image_names] if preload else image_names
        for i in range(0, n_centers, batch_size):
            print('%f%% tested (step %d)' % (100.0*i/n_centers, (i/batch_size)+1), end='\r')
            sys.stdout.flush()
            x = get_patches_list([image_list], [centers[i:i + batch_size]], size, preload)
            x = np.concatenate(x).astype(dtype=datatype)
            yield x


def load_masks(mask_names):
    for image_name in mask_names:
        yield load_nii(image_name).get_data().astype(dtype=np.bool)


def get_cnn_centers(names, labels_names, balanced=True, neigh_width=15):
    rois = list(load_masks(names))
    rois_p = list(load_masks(labels_names))
    rois_p_neigh = [log_and(log_and(imdilate(roi_p, iterations=neigh_width), log_not(roi_p)), roi)
                    for roi, roi_p in izip(rois, rois_p)]
    rois_n_global = [log_and(roi, log_not(log_or(roi_pn, roi_p)))
                     for roi, roi_pn, roi_p in izip(rois, rois_p_neigh, rois_p)]
    rois = list()
    for roi_pn, roi_ng, roi_p in izip(rois_p_neigh, rois_n_global, rois_p):
        # The goal of this for is to randomly select the same number of nonlesion and lesion samples for each image.
        # We also want to make sure that we select the same number of boundary negatives and general negatives to
        # try to account for the variability in the brain.
        n_positive = np.count_nonzero(roi_p)
        if balanced:
            roi_pn[roi_pn] = np.random.permutation(xrange(np.count_nonzero(roi_pn))) < n_positive / 2
        roi_ng[roi_ng] = np.random.permutation(xrange(np.count_nonzero(roi_ng))) < n_positive / 2
        rois.append(log_or(log_or(roi_ng, roi_pn), roi_p))

    # In order to be able to permute the centers to randomly select them, or just shuffle them for training, we need
    # to keep the image reference with the center. That's why we are doing the next following lines of code.
    for roi in rois:
        print(np.count_nonzero(roi))
    centers_list = [get_mask_voxels(roi) for roi in rois]
    print(centers_list)
    idx_lesion_centers = np.concatenate([np.array([(i, c) for c in centers], dtype=object)
                                         for i, centers in enumerate(centers_list)])

    return idx_lesion_centers
