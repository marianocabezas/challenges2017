from operator import itemgetter
import numpy as np
from nibabel import load as load_nii
from data_manipulation.generate_features import get_mask_voxels, get_patches, get_patches2_5d
from itertools import izip, chain
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


def get_image_patches(list_of_image_lists, center_list, size):
    return [np.stack([get_patches(image, centers, size) for image in image_list], axis=1)
            for image_list, centers in izip(list_of_image_lists, center_list)]


def get_norm_patches(image_names, centers, size):
    return [get_patches(norm(load_nii(name).get_data()), centers, size) for name in image_names]


def get_stacked_patches(image_name_list, centers_list, size):
    patches = [np.stack(get_norm_patches(image_names, centers, size), axis=1)
               for image_names, centers in izip(image_name_list, centers_list) if centers]

    return patches


def centers_and_idx(centers, n_images):
    centers = [list(map(lambda z: tuple(z[1]) if z[0] == im else [], centers)) for im in range(n_images)]
    idx = [map(lambda (a, b): [a] if b else [], enumerate(c)) for c in centers]
    centers = [filter(bool, c) for c in centers]
    idx = list(chain(*[chain(*i) for i in idx]))
    return centers, idx


def images_norm_generator(image_names):
    for patient in image_names:
        images = [load_nii(image_name).get_data() for image_name in patient]
        yield [norm(im) for im in images]


def labels_generator(image_names):
    for patient in image_names:
        yield load_nii(patient).get_data()


def load_patch_batch(
        image_names,
        label_names,
        centers,
        batch_size,
        size,
        nlabels,
        datatype=np.float32,
        preload=False
):
    while True:
        gen = load_patch_batch_generator_preload(
            image_names=image_names,
            label_names=label_names,
            centers=centers,
            batch_size=batch_size,
            size=size,
            nlabels=nlabels,
            datatype=datatype,
        ) if preload else load_patch_batch_generator(
            image_names=image_names,
            label_names=label_names,
            centers=centers,
            batch_size=batch_size,
            size=size,
            nlabels=nlabels,
            datatype=datatype,
        )
        for x, y in gen:
            yield x, y


def load_patch_batch_generator(
            image_names,
            label_names,
            centers,
            batch_size,
            size,
            nlabels,
            datatype=np.float32
):
    n_centers = len(centers)
    n_images = len(image_names)
    for i in range(0, n_centers, batch_size):
        centers, idx = centers_and_idx(centers[i:i + batch_size], n_images)
        x = get_stacked_patches(image_names, centers, size)
        x = np.concatenate(filter(lambda z: z.any(), x)).astype(dtype=datatype)
        x[idx] = x
        y = [np.array([l[c] for c in lc]) for l, lc in izip(labels_generator(label_names), centers)]
        y = np.concatenate(y)
        y[idx] = y
        yield (x, keras.utils.to_categorical(y, num_classes=nlabels))


def load_patch_batch_generator_preload(
        image_names,
        label_names,
        centers,
        batch_size,
        size,
        nlabels,
        datatype=np.float32
):
    n_centers = len(centers)
    n_images = len(image_names)
    images = [images_norm_generator(image_names)]
    for i in range(0, n_centers, batch_size):
        centers, idx = centers_and_idx(centers[i:i + batch_size], n_images)
        x = get_image_patches(images, centers, size)
        x = np.concatenate(filter(lambda z: z.any(), x)).astype(dtype=datatype)
        x[idx] = x
        y = [np.array([l[c] for c in lc]) for l, lc in izip(labels_generator(label_names), centers)]
        y = np.concatenate(y)
        y[idx] = y
        yield (x, keras.utils.to_categorical(y, num_classes=nlabels))


def get_centers_from_masks(positive_masks, negative_masks, balanced=True, random_state=42):
    positive_centers = [get_mask_voxels(mask) for mask in positive_masks]
    negative_centers = [get_mask_voxels(mask) for mask in negative_masks]
    if balanced:
        positive_voxels = [len(positives) for positives in positive_centers]
        negative_centers = list(subsample(negative_centers, positive_voxels, random_state))

    return positive_centers, negative_centers


def load_masks(mask_names):
    for image_name in mask_names:
        yield load_nii(image_name).get_data().astype(dtype=np.bool)


def get_cnn_centers(names, labels_names, neigh_width=15, dfactor=10):
    rois = load_masks(names)
    rois_p = list(load_masks(labels_names))
    rois_p_neigh = [log_and(imdilate(roi_p, iterations=neigh_width), log_not(roi_p))
                    for roi_p in rois_p]
    rois_n_global = [log_and(roi, log_not(log_or(roi, roi_p)))
                     for roi, roi_pn, roi_p in izip(rois, rois_p_neigh, rois_p)]
    rois = list()
    for roi_pn, roi_ng, roi_p in izip(rois_p_neigh, rois_n_global, rois_p):
        # Using the Python trick for "ceil" with integers
        roi_pn[roi_pn] = np.random.permutation(xrange(np.count_nonzero(roi_pn))) < np.count_nonzero(roi_p)
        roi_ng[roi_ng] = np.random.permutation(xrange(np.count_nonzero(roi_ng))) < np.count_nonzero(roi_p)
        rois.append(log_or(log_or(roi_ng, roi_pn), roi_p))

    centers_list = [get_mask_voxels(roi) for roi in rois]
    idx_lesion_centers = np.concatenate([np.array([(i, c) for c in centers], dtype=object)
                                         for i, centers in enumerate(centers_list)])

    return np.random.permutation(idx_lesion_centers[::dfactor])
