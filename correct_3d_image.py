from utils import *
import scipy.ndimage.morphology as morph
import scipy.ndimage.filters as filters
from scipy.ndimage.measurements import label as find_cc


def dilate_label(img, label, r):
    img_labels = one_hot(img)
    img_labels[..., label] = 2 * morph.binary_dilation(
        img_labels[..., label], sphere(r)).astype(img_labels.dtype)
    img = np.argmax(img_labels, axis=-1).astype(np.uint8)
    return img


def count_neighbors(img_labels, r):
    neighbors_count = np.zeros_like(img_labels, dtype=np.int32)
    for i in range(img_labels.shape[-1]):
        neighbors_count[..., i] = filters.correlate(
            img_labels[..., i], sphere(r), mode='constant', cval=0)
    return neighbors_count


def smooth(img, r):
    img_labels = one_hot(img)
    return np.argmax(count_neighbors(img_labels, r), axis=-1).astype(np.uint8)


def replace_small_cc(img, r, cc_thresh):
    # TODO it may be not correct
    bad_pixels_prev = np.inf
    while True:
        img_labels = one_hot(img)
        small_cc_mask = np.zeros_like(img, np.bool)
        for label in range(img_labels.shape[-1]):
            cc_ids, cc_num = find_cc(img_labels[..., label])
            print('Material:', label, 'cc:', cc_num)
            for cc_id in range(1, cc_num + 1):
                cc_mask = cc_ids == cc_id
                if np.sum(cc_mask) < cc_thresh:
                    small_cc_mask[cc_mask] = True

        bad_pixels = np.sum(small_cc_mask)
        print('Bad pixels count:', bad_pixels)
        if bad_pixels == bad_pixels_prev:
            print('-' * 40)
            return img
        bad_pixels_prev = bad_pixels

        img_labels[small_cc_mask, :] = 0
        neighbors = count_neighbors(img_labels, r)
        img[small_cc_mask] = np.argmax(neighbors[small_cc_mask], axis=-1)


img = read_vti('grids/img.vti')
print('a')
img = dilate_label(img, 5, 4)
print('b')

img = replace_small_cc(img, 4, 400)
print('c')
img = smooth(img, 4)
print('d')
img = replace_small_cc(img, 4, 400)
print('e')

write_vti(img, 'grids/img_smoothed.vti')
print('f')
write_inrimage(img, 'grids/img.inr', 200 / img.shape[0])
print('g')

# next call the CGAL mesher https://doc.cgal.org/latest/Mesh_3/index.html#title25

# to_vtk('grids/out.mesh')
# ps, cs, _ = convertVtkGridToNumpy(readVtkGrid('grids/out.vtk'), 4)
# grid_quality(ps, cs)











