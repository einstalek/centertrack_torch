import random

import cv2
import numpy as np


def crop_near_box(img, box, anns, scale=1., shift=0.5, enl=(1.5, 3.)):
    h, w, _ = img.shape
    # randomly shift and enlarge box
    x1, y1, bw, bh = box
    cx, cy = x1 + bw / 2, y1 + bh / 2
    cx = cx + bw * np.random.uniform(-shift, shift)
    cy = cy + bh * np.random.uniform(-shift, shift)
    bw = bw * np.random.uniform(*enl)
    bh = bh * np.random.uniform(*enl)
    x1, y1 = cx - scale * bw, cy - scale * bh
    x2, y2 = cx + scale * bw, cy + scale * bh
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w-1, x2), min(h-1, y2)
    x1, y1, x2, y2 = list(map(int, (x1, y1, x2, y2)))
    crop = img[y1:y2, x1:x2, :]
    h, w, _ = crop.shape
    new_anns = []
    # apply transform to the rest of annotations
#     for ann in deepcopy(anns):
    for ann in anns:
        box = ann['bbox']
        box = [box[0]-x1, box[1]-y1, box[2], box[3]]
        cx, cy = box[0] + box[2]/2, box[1] + box[3]/2
        if not (0 < cx < w-1 and 0 < cy < h-1):
            continue
        box[0], box[1] = max(0, box[0]), max(0, box[1])
        ann['bbox'] = box
        new_anns.append(ann)
    return crop, new_anns


def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def get_resize_transform(in_h, in_w, out_h, out_w):
    src = np.array([[0, 0], [0, in_h], [in_w, 0]], dtype=np.float32)
    dst = np.array([[0, 0], [0, out_h], [out_w, 0]], dtype=np.float32)
    trans = cv2.getAffineTransform(src, dst)
    return trans


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs
    return src_result


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]

# @numba.jit(nopython=True, nogil=True)
def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


# @numba.jit(nopython=True, nogil=True)
def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    return h


# @numba.jit(nopython=True, nogil=True)
def draw_umich_gaussian(heatmap, center, radius, k=1):
    # import pdb; pdb.set_trace()
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 4)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)
    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def color_aug(data_rng, image, eig_val, eig_vec):
    functions = [brightness_, contrast_, saturation_]
    random.shuffle(functions)

    gs = grayscale(image)
    gs_mean = gs.mean()
    for f in functions:
        f(data_rng, image, gs, gs_mean, 0.4)
    lighting_(data_rng, image, 0.1, eig_val, eig_vec)


def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def lighting_(data_rng, image, alphastd, eigval, eigvec):
    alpha = data_rng.normal(scale=alphastd, size=(3,))
    image += np.dot(eigvec, eigval * alpha)


def blend_(alpha, image1, image2):
    image1 *= alpha
    image2 *= (1 - alpha)
    image1 += image2


def saturation_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    blend_(alpha, image, gs[:, :, None])


def brightness_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    image *= alpha


def contrast_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    blend_(alpha, image, gs_mean)


def gamma_transform(args, inp):
    inp = inp ** np.random.uniform(*args.gamma)
    return inp


def brightness_transform(_xx, _yy, inp):
    xx, yy = _xx, _yy
    alpha = np.random.uniform(0.3, 0.7)
    gamma = np.random.uniform(-1, 5)
    beta_x = np.random.uniform(-2., 2.)
    beta_y = np.random.uniform(-2., 2.)
    zz = alpha + np.abs(beta_y * yy + beta_x * xx ** gamma) + np.random.normal(scale=0.05, size=xx.shape)
    zz = np.clip(zz, 0, 1)
    zz = cv2.medianBlur(zz.astype(np.float32), 5, 5)
    return inp * zz[..., None]