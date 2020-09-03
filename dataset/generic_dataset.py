import copy
import math
import random
import os
from collections import defaultdict

import cv2
import torch.utils.data as data
import numpy as np
from pycocotools import coco

from dataset.utils import get_affine_transform, affine_transform, crop_near_box, \
    draw_umich_gaussian, gaussian_radius, get_resize_transform, gamma_transform, brightness_transform

from dataset.augmentations import transform_fn


class GenericDataset(data.Dataset):
    flip = 0.4
    same_aug_pre = True
    num_classes = 1

    is_fusion_dataset = False
    default_resolution = [256, 256]
    num_categories = 1
    class_name = 'human'
    cat_ids = {1: 1}
    rest_focal_length = 1200
    num_joints = 17
    ignore_val = 1

    def __init__(self, args, ann_path=None, img_dir=None, split='train', group_rates=None):
        super(GenericDataset, self).__init__()
        self.split = split
        self.num_frames = args.num_frames
        self.aug_rot = args.aug_rot
        self.rotate = args.rotate
        self.aug_s = args.aug_s
        self.pre_hm = args.pre_hm
        self.max_objs = args.max_objs
        self.no_color_aug = args.no_color_aug
        self.max_frame_dist = args.max_frame_dist
        self.heads = args.heads
        self.down_ratio = args.down_ratio
        self.fp_disturb = args.fp_disturb
        self.lost_disturb = args.lost_disturb
        self.hm_disturb = args.hm_disturb
        self.dense_reg = 1
        self.input_h = args.input_h
        self.input_w = args.input_w
        self.output_h = self.input_h // self.down_ratio
        self.output_w = self.input_w // self.down_ratio
        self._data_rng = np.random.RandomState(123)
        self.args = args
        if ann_path is not None and img_dir is not None:
            print('==> initializing data from {}, \n images from {} ...'.format(ann_path, img_dir))
            self.coco = coco.COCO(ann_path)
            self.images = self.coco.getImgIds()
            self.img_dir = img_dir

            if not ('videos' in self.coco.dataset):
                self.fake_video_data()
            self.video_to_images = defaultdict(list)
            for image in self.coco.dataset['images']:
                self.video_to_images[image['video_id']].append(image)
            if self.split == 'train':
                self._group_indices_by_dataset(group_rates)
        self.__len = 0
        self._xx = 1 + np.arange(self.input_w)
        self._xx = np.tile(self._xx, [self.input_h, 1]) / self.input_w
        self._yy = 1 + np.arange(self.input_h)
        self._yy = np.tile(self._yy, [self.input_w, 1]).T / self.input_h

    def fake_video_data(self):
        self.coco.dataset['videos'] = []
        for i in range(len(self.coco.dataset['images'])):
            img_id = self.coco.dataset['images'][i]['id']
            self.coco.dataset['images'][i]['video_id'] = img_id
            self.coco.dataset['images'][i]['frame_id'] = 1
            self.coco.dataset['videos'].append({'id': img_id})

    def _group_indices_by_dataset(self, group_rates):
        group_indices = {k: set() for k in group_rates}
        for x in self.coco.imgs.values():
            for group in group_rates:
                found = False
                for key in group:
                    if key in x['file_name']:
                        group_indices[group].add(x['id'])
                        found = True
                if found:
                    continue
        self.group_indices = group_indices
        self.group_probas = np.array([group_rates[k] / self.args.batch_size for k in group_rates])

    def _load_image_anns(self, img_id, coco, img_dir):
        img_info = coco.loadImgs(ids=[img_id])[0]
        file_name = img_info['file_name']
        img_path = os.path.join(img_dir, file_name)
        ann_ids = coco.getAnnIds(imgIds=[img_id])
        anns = copy.deepcopy(coco.loadAnns(ids=ann_ids))
        if '.npy' in img_path:
            img = np.load(img_path)
        else:
            assert os.path.exists(img_path), img_path
            img = cv2.imread(img_path)

            if self.args.cvt_gray:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = np.tile(img[..., None], 3)

        img, anns = self._pad_image_boxes(img, anns)
        return img[..., ::-1], anns, img_info, img_path

    def _pad_image_boxes(self, img, anns, size=(384, 224)):
        "To convert image to 384x224 we need w=1.71h"
        h, w, _ = img.shape
        if h > (1 / 1.71) * w:
            pad = int(1.71 * h - w)
            img = np.pad(img, ((0, 0), (pad // 2, pad // 2), (0, 0)))
            img = cv2.resize(img, size)
            scale_x, scale_y = 1.71 * h / size[0], h / size[1]
            for ann in anns:
                x1, y1, w, h = ann['bbox']
                x1 += pad / 2
                x1 /= scale_x
                y1 /= scale_y
                w /= scale_x
                h /= scale_y
                ann['bbox'] = [x1, y1, w, h]
        return img, anns

    def _get_augm_image(self, img):
        try:
            img = transform_fn(image=img)['image']
        except:
            pass
        return img

    def _load_data(self, index):
        img_dir = self.img_dir
        img_id = self.images[index]
        img, anns, img_info, img_path = self._load_image_anns(img_id, self.coco, img_dir)
        return img, anns, img_info, img_path

    def _load_pre_data(self, video_id, frame_id, static, img_id, num_frames=1):
        prev_frames = []
        if static:
            frame_dist = 0
            img, anns, _, img_path = self._load_image_anns(img_id, self.coco, self.img_dir)
            if num_frames > 1:
                prev_frames = [img] * (num_frames - 1)
        elif self.split == "train":
            if np.random.rand() < self.args.max_dist_proba:
                max_frame_dist = np.random.randint(low=1, high=self.max_frame_dist) + int(np.random.exponential(10))
            else:
                max_frame_dist = np.random.randint(low=1, high=4)
            img_infos = self.video_to_images[video_id]
            img_ids = [(img_info['id'], img_info['frame_id']) \
                       for img_info in img_infos \
                       if abs(img_info['frame_id'] - frame_id) < max_frame_dist]
            rand_id = np.random.choice(len(img_ids))
            img_id, pre_frame_id = img_ids[rand_id]
            frame_dist = abs(frame_id - pre_frame_id)
            img, anns, _, img_path = self._load_image_anns(img_id, self.coco, self.img_dir)
            if num_frames > 1:
                for rand in random.sample(img_ids, num_frames - 1):
                    img_id, pre_frame_id = rand
                    prev_frame, _, _, _ = self._load_image_anns(img_id, self.coco, self.img_dir)
                    prev_frames.append(prev_frame)
        else:
            # for validation get previous frame
            img_infos = self.video_to_images[video_id]
            img_ids = [(img_info['id'], img_info['frame_id']) \
                       for img_info in img_infos \
                       if frame_id - img_info['frame_id'] == self.args.val_skip_frame]
            if len(img_ids) == 0:
                frame_dist = 0
                img, anns, _, img_path = self._load_image_anns(img_id, self.coco, self.img_dir)
            else:
                img_id, pre_frame_id = img_ids[0]
                img, anns, _, img_path = self._load_image_anns(img_id, self.coco, self.img_dir)
                frame_dist = self.args.val_skip_frame
        return img, anns, frame_dist, prev_frames, img_path

    def __getitem__(self, index):
        if self.split == "train":
            _group = np.random.choice(
                list(self.group_indices.keys()), p=self.group_probas
            )
            index = random.sample(self.group_indices[_group], 1)[0]
            index = min(len(self) - 1, index)

        img, anns, img_info, img_path = self._load_data(index)
        height, width, *_ = img.shape

        cropped, crop_id = False, None
        if np.random.random() < self.args.crop_near_box and len(anns) > 0:
            _ann = [x for x in anns if x['conf'] > 0]
            if len(_ann) > 0:
                _ann = random.sample(_ann, 1)[0]
                box = _ann['bbox']
                if box[2] > self.args.crop_min_box_size and box[3] > self.args.crop_min_box_size:
                    img, anns = crop_near_box(img, box, anns)
                    img, anns = self._pad_image_boxes(img, anns)
                    cropped, crop_id = True, _ann['track_id']

        static = (img_info['prev_image_id'] == -1 and img_info['next_image_id'] == -1)
        height, width, *_ = img.shape
        if len(img.shape) == 2:
            img = img[..., None]
        c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
        s = max(height, width) * 1.0
        aug_s, rot, flipped = 1, 0, 0

        c, aug_s, rot = self._get_aug_param(c, s, width, height, static=static)
        s = s * aug_s

        if self.split == 'train' and np.random.random() < self.flip:
            flipped = 1
            img = img[:, ::-1, :]
            anns = self._flip_anns(anns, width)

        if self.split == 'train':
            trans_input = get_affine_transform(c, s, rot, [self.input_w, self.input_h])
            trans_output = get_affine_transform(c, s, rot, [self.output_w, self.output_h])
        else:
            trans_input = get_resize_transform(height, width, self.input_h, self.input_w)
            trans_output = get_resize_transform(height, width, self.output_h, self.output_w)

        inp = self._get_input(img, trans_input)
        ret = {'image': inp}
        if self.args.ret_fpath:
            ret['fpath'] = img_path
        gt_det = {'bboxes': [], 'scores': [], 'clses': [], 'cts': []}

        pre_cts, track_ids = None, None

        pre_image, pre_anns, frame_dist, prev_frames, prev_fpath = self._load_pre_data(
            img_info['video_id'], img_info['frame_id'],
            static, img_info['id'], self.num_frames)
        height, width, *_ = pre_image.shape
        if self.args.ret_fpath:
            ret['prev_fpath'] = prev_fpath

        if cropped:
            _ann = [x for x in pre_anns if x['track_id'] == crop_id]
            if len(_ann) == 1:
                box = _ann[0]['bbox']
                pre_image, pre_anns = crop_near_box(pre_image, box, pre_anns, enl=(2.5, 3.))
                pre_image, pre_anns = self._pad_image_boxes(pre_image, pre_anns)
        height, width, *_ = pre_image.shape

        if len(pre_image.shape) == 2:
            pre_image = pre_image[..., None]
        if self.split == 'train' and flipped:
            pre_image = pre_image[:, ::-1, :].copy()
            pre_anns = self._flip_anns(pre_anns, width)
            for i, pic in enumerate(prev_frames):
                prev_frames[i] = pic[:, ::-1, :]

        if self.split == "train" and self.same_aug_pre and frame_dist != 0:
            trans_input_pre = trans_input
            trans_output_pre = trans_output
        elif self.split == "train":
            c_pre, aug_s_pre, _ = self._get_aug_param(
                c, s, width, height, disturb=True, static=static)
            aug_s_pre = aug_s
            ######### TODO: uncomment if needed #########
            s_pre = s * aug_s_pre
            trans_input_pre = get_affine_transform(
                c_pre, s_pre, rot, [self.input_w, self.input_h])
            trans_output_pre = get_affine_transform(
                c_pre, s_pre, rot, [self.output_w, self.output_h])
        else:
            trans_input_pre = get_resize_transform(height, width, self.input_h, self.input_w)
            trans_output_pre = get_resize_transform(height, width, self.output_h, self.output_w)

        pre_img = self._get_input(pre_image, trans_input_pre)
        for i, pic in enumerate(prev_frames):
            prev_frames[i] = self._get_input(pic, trans_input_pre)
        pre_hm, pre_cts, track_ids = self._get_pre_dets(pre_anns, trans_input_pre, trans_output_pre)
        ret['pre_img'] = pre_img
        ret['prev_frames'] = prev_frames
        if self.pre_hm:
            ret['pre_hm'] = pre_hm

        self._init_ret(ret, gt_det)
        calib = self._get_calib(img_info, width, height)

        num_objs = min(len(anns), self.max_objs)
        for k in range(num_objs):
            ann = anns[k]
            cls_id = int(self.cat_ids[ann['category_id']])
            if cls_id > self.num_classes or cls_id <= -999:
                continue
            bbox, bbox_amodal = self._get_bbox_output(
                ann['bbox'], trans_output, height, width)
            self._add_instance(
                ret, gt_det, k, cls_id, bbox, bbox_amodal, ann, trans_output, aug_s,
                calib, pre_cts, track_ids)
        return ret

    def _get_aug_param(self, c, s, width, height, disturb=False, static=False):
        aug_s = np.random.choice(np.arange(*self.aug_s, 0.1))
        w_border = self._get_border(512, width)
        h_border = self._get_border(512, height)
        c[0] = np.random.randint(low=w_border, high=width - w_border)
        c[1] = np.random.randint(low=h_border, high=height - h_border)

        if np.random.random() < self.aug_rot and self.split == 'train':
            rf = self.rotate
            rot = np.clip(np.random.randn() * rf, -rf * 2, rf * 2)
        else:
            rot = 0
        return c, aug_s, rot

    @staticmethod
    def _get_border(border, size):
        i = 1
        while size - border // i <= border // i:
            i *= 2
        return border // i

    @staticmethod
    def _flip_anns(anns, width):
        for k in range(len(anns)):
            bbox = anns[k]['bbox']
            anns[k]['bbox'] = [
                width - bbox[0] - 1 - bbox[2], bbox[1], bbox[2], bbox[3]]
        return anns

    def _get_input(self, img, trans_input):
        inp = cv2.warpAffine(img, trans_input,
                             (self.input_w, self.input_h),
                             flags=cv2.INTER_LINEAR)
        if self.split == 'train' and not self.no_color_aug:
            try:
                inp = transform_fn(image=inp.astype(np.uint8))['image']
            except:
                pass
        inp = (inp.astype(np.float32) / 255.)
        if self.split == "train" and self.args.use_gamma:
            seed = np.random.random()
            if seed < 0.33:
                inp = gamma_transform(self.args, inp)
            elif seed < 0.66:
                inp = brightness_transform(self._xx, self._yy, inp)
        inp = (inp - 0.5) / 0.5
        inp = inp.transpose(2, 0, 1)
        return inp

    def _get_pre_dets(self, anns, trans_input, trans_output):
        hm_h, hm_w = self.input_h, self.input_w
        down_ratio = self.down_ratio
        trans = trans_input
        return_hm = self.pre_hm
        pre_hm = np.zeros((1, hm_h, hm_w), dtype=np.float32) if return_hm else None
        pre_cts, track_ids = [], []
        for ann in anns:
            cls_id = int(self.cat_ids[ann['category_id']])
            if cls_id > self.num_classes or cls_id <= -99 or \
                    ('iscrowd' in ann and ann['iscrowd'] > 0):
                continue
            bbox = self._coco_box_to_bbox(ann['bbox'])
            bbox[:2] = affine_transform(bbox[:2], trans)
            bbox[2:] = affine_transform(bbox[2:], trans)
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, hm_w - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, hm_h - 1)
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            max_rad = 1
            if h > 0 and w > 0:
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(self.args.pre_hm_min_radius, int(radius))
                max_rad = max(max_rad, radius)
                ct = np.array(
                    [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                ct0 = ct.copy()
                conf = 1

                ct[0] = ct[0] + np.random.randn() * self.hm_disturb * w
                ct[1] = ct[1] + np.random.randn() * self.hm_disturb * h
                conf = 1 if np.random.random() > self.lost_disturb else 0

                ct_int = ct.astype(np.int32)
                if conf == 0:
                    pre_cts.append(ct / down_ratio)
                else:
                    pre_cts.append(ct0 / down_ratio)

                track_ids.append(ann['track_id'] if 'track_id' in ann else -1)
                if return_hm:
                    draw_umich_gaussian(pre_hm[0], ct_int, radius, k=conf)

                if np.random.random() < self.fp_disturb and return_hm:
                    ct2 = ct0.copy()
                    # Hard code heatmap disturb ratio, haven't tried other numbers.
                    ct2[0] = ct2[0] + np.random.randn() * 0.05 * w
                    ct2[1] = ct2[1] + np.random.randn() * 0.05 * h
                    ct2_int = ct2.astype(np.int32)
                    draw_umich_gaussian(pre_hm[0], ct2_int, radius, k=conf)
        return pre_hm, pre_cts, track_ids

    @staticmethod
    def _coco_box_to_bbox(box):
        bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                        dtype=np.float32)
        return bbox

    def _init_ret(self, ret, gt_det):
        max_objs = self.max_objs * self.dense_reg
        ret['hm'] = np.zeros(
            (self.num_classes, self.output_h, self.output_w),
            np.float32)
        ret['ind'] = np.zeros(max_objs, dtype=np.int64)
        ret['cat'] = np.zeros(max_objs, dtype=np.int64)
        ret['mask'] = np.zeros(max_objs, dtype=np.float32)

        regression_head_dims = {
            'reg': 2, 'wh': 2, 'tracking': 2, 'ltrb': 4, 'ltrb_amodal': 4,
            'nuscenes_att': 8, 'velocity': 3, 'hps': self.num_joints * 2,
            'dep': 1, 'dim': 3, 'amodel_offset': 2}

        for head in regression_head_dims:
            if head in self.heads:
                ret[head] = np.zeros(
                    (max_objs, regression_head_dims[head]), dtype=np.float32)
                ret[head + '_mask'] = np.zeros(
                    (max_objs, regression_head_dims[head]), dtype=np.float32)
                gt_det[head] = []

    def _get_calib(self, img_info, width, height):
        if 'calib' in img_info:
            calib = np.array(img_info['calib'], dtype=np.float32)
        else:
            calib = np.array([[self.rest_focal_length, 0, width / 2, 0],
                              [0, self.rest_focal_length, height / 2, 0],
                              [0, 0, 1, 0]])
        return calib

    def _get_bbox_output(self, bbox, trans_output, height, width):
        bbox = self._coco_box_to_bbox(bbox).copy()

        rect = np.array([[bbox[0], bbox[1]], [bbox[0], bbox[3]],
                         [bbox[2], bbox[3]], [bbox[2], bbox[1]]], dtype=np.float32)
        for t in range(4):
            rect[t] = affine_transform(rect[t], trans_output)
        bbox[:2] = rect[:, 0].min(), rect[:, 1].min()
        bbox[2:] = rect[:, 0].max(), rect[:, 1].max()

        bbox_amodal = copy.deepcopy(bbox)
        bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, self.output_w - 1)
        bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, self.output_h - 1)
        return bbox, bbox_amodal

    def _add_instance(
            self, ret, gt_det, k, cls_id, bbox, bbox_amodal, ann, trans_output,
            aug_s, calib, pre_cts=None, track_ids=None):
        if ann['conf'] < 1.0:
            return
        h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
        h *= self.args.widen_boxes
        w *= self.args.widen_boxes
        if h <= 0 or w <= 0:
            return
        radius = gaussian_radius((math.ceil(h), math.ceil(w)))
        # set lower limit on the gaussian radius to 2
        radius = max(self.args.gaussian_min_radius, int(radius))
        ct = np.array(
            [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
        ct_int = ct.astype(np.int32)
        ret['cat'][k] = cls_id - 1
        ret['mask'][k] = 1
        if 'wh' in ret:
            ret['wh'][k] = 1. * w, 1. * h
            ret['wh_mask'][k] = 1

        ret['ind'][k] = ct_int[1] * self.output_w + ct_int[0]
        ret['reg'][k] = ct - ct_int
        ret['reg_mask'][k] = 1
        draw_umich_gaussian(ret['hm'][cls_id - 1], ct_int, radius)

        gt_det['bboxes'].append(
            np.array([ct[0] - w / 2, ct[1] - h / 2,
                      ct[0] + w / 2, ct[1] + h / 2], dtype=np.float32))
        gt_det['scores'].append(1)
        gt_det['clses'].append(cls_id - 1)
        gt_det['cts'].append(ct)

        if 'tracking' in self.heads:
            if ann['track_id'] in track_ids:
                pre_ct = pre_cts[track_ids.index(ann['track_id'])]
                ret['tracking_mask'][k] = 1
                ret['tracking'][k] = pre_ct - ct_int
                gt_det['tracking'].append(ret['tracking'][k])
            else:
                gt_det['tracking'].append(np.zeros(2, np.float32))

    def __len__(self):
        if self.__len == 0:
            self.__len = len(self.images)
        return self.__len
