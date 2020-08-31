import horovod.torch as hvd

hvd.init()

import os
import sys

sys.path.insert(0, '/home/jovyan/env/lib/python3.6/site-packages/')

import torch

torch.cuda.set_device(hvd.local_rank())

from tracker.tracker import Tracker
from args import Args
import numpy as np
import cv2

from blazepalm.detector import BlazePalm
from trainer.utils import generic_decode


args = Args()
model = BlazePalm(args)
ckpt = 46
state_dict = torch.load("/home/jovyan/CenterTrack/weights_1/model_{}.pth".format(ckpt))
model.load_state_dict(state_dict['state_dict'], strict=True)
model.to(args.device)

fps = os.listdir("/home/jovyan/mAP/input/ground-truth-ct/")
fps = [x.split('.')[0] for x in fps]


def get_input(img, size=(384, 384)):
    h, w, _ = img.shape
    pad = max(h, w) - min(h, w)
    img = cv2.resize(img, size)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.tile(img[..., None], 3)

    img = (img.astype(np.float32) - 127.5) / 127.5
    img = img.transpose(2, 0, 1)
    return img, (h, w), pad


skip_frame = 2
save = '/home/jovyan/mAP/input/detection-results-ct/'
roots = ('/home/jovyan/dataset/0002_navigation_gestures_abc_home/frames_holdout/',
         '/home/jovyan/dataset/0004_static_gestures_abc_home/frames_holdout/',
         '/home/jovyan/dataset/0010_youtube_active_gesticulation/frames_holdout/',)

tracker = None
curr_vid = None
size = (384, 224)
for i in range(len(fps)):
    root = None
    for r in roots:
        if os.path.exists(os.path.join(r, fps[i] + '.jpg')):
            root = r
            break
    if root is None:
        print(fps[i])
        continue

    _pic = cv2.imread(os.path.join(root, fps[i] + '.jpg'))[..., ::-1]
    save_fp = os.path.join(save, fps[i] + ".txt")
    pic, (s_h, s_w), pad = get_input(_pic, size)
    s_h, s_w = s_h / size[1], s_w / size[0]

    num = int(fps[i].split('_')[-1])
    vid_name = '_'.join(fps[i].split('_')[:-1])

    if vid_name != curr_vid:
        curr_vid = vid_name
        tracker = Tracker(args)

    if num % 10 != 0:
        continue

    prev_frame_fp = None
    prev_frame = None
    if num > 1:
        prev_frame_fp = "_".join(fps[i].split("_")[:-1]) + "_" + str(num - skip_frame)
    if prev_frame_fp is not None and os.path.exists(os.path.join(root, prev_frame_fp + '.jpg')):
        _pic = cv2.imread(os.path.join(root, prev_frame_fp + '.jpg'))[..., ::-1]
        prev_frame, (s_h, s_w), pad = get_input(_pic, size)
        s_h, s_w = s_h / size[1], s_w / size[0]

    image = torch.from_numpy(pic[np.newaxis])
    image = image.to(args.device, non_blocking=True)

    if prev_frame is None:
        pre_image = torch.from_numpy(pic[np.newaxis])
        pre_image = pre_image.to(args.device, non_blocking=True)
    else:
        pre_image = torch.from_numpy(prev_frame[np.newaxis])
        pre_image = pre_image.to(args.device, non_blocking=True)

    with torch.no_grad():
        out = model(torch.cat([image, pre_image], axis=1))

    res = []
    for i, (head, x) in enumerate(zip(args.heads.keys(), out)):
        if head in ('hm',):
            res.append(x.sigmoid_())
        else:
            res.append(x)
    dets = generic_decode({k: res[i] for (i, k) in enumerate(args.heads)}, 10, args)
    for k in dets:
        dets[k] = dets[k].detach().cpu().numpy()

    if not tracker.init and len(dets) > 0:
        tracker.init_track(dets)
    elif len(dets) > 0:
        tracker.step(dets)

    with open(save_fp, "w") as f:
        for track in tracker.tracks:
            x1, y1, x2, y2 = args.down_ratio * track['bbox']
            x1, x2 = x1 * s_w, x2 * s_w
            y1, y2 = s_h * y1, s_h * y2
            score = track['score']
            f.write("{} {} {} {} {} {}\n".format(score,
                                                 track['tracking_id'], x1, y1, x2, y2))
