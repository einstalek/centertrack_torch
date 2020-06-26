import os
import random
import numpy as np
import json

DATA_PATH = '/home/jovyan/dataset/'
OUT_PATH = os.path.join(DATA_PATH, '')
split = "train"

def get_num(fp):
    return int(fp.split('_')[-1].split('.')[0])


if __name__ == '__main__':
    out_path = os.path.join(OUT_PATH, '{}.json'.format(split))
    out = {'images': [], 'annotations': [],
           'categories': [{'id': 1, 'name': 'human'}],
           'videos': []}

    image_cnt = 0
    ann_cnt = 0
    video_cnt = 0

    folders = ("0002_navigation_gestures_abc_home/",
               "0004_static_gestures_abc_home/",
               "0010_youtube_active_gesticulation/")

    for folder in folders:
        seqs = [x.split('.')[0] for x in os.listdir("./{}/mot/".format(folder))]
        seqs = [x for x in seqs if x]
        random.seed(10)
        val = random.sample(seqs, 3)
        train = [x for x in seqs if x not in val]
        sample = train if split == 'train' else val

        seq_fp = "{}/frames".format(folder)
        seq_path = os.path.join(DATA_PATH, seq_fp)
        images = os.listdir(seq_path)
        for seq in sample:
            if '.DS_Store' in seq:
                continue
            video_cnt += 1
            out['videos'].append({'id': video_cnt, 'file_name': seq})
            _images = [get_num(x) for x in images if seq + '_' in x]
            assert len(_images) > 0
            _min, _max = min(_images), max(_images)
            num_images = len(_images)
            ann_path = os.path.join(DATA_PATH, folder, "mot", "{}.txt".format(seq))

            anns = np.loadtxt(ann_path, dtype=np.float32, delimiter=',')

            for i, num in enumerate(range(_min, _max + 1)):
                image_info = {'file_name': '{}/{}_frame_{}.jpg'.format(seq_fp, seq, num),
                              'id': image_cnt + i + 1,
                              'frame_id': i + 1,
                              'prev_image_id': image_cnt + i if i > 0 else -1,
                              'next_image_id': image_cnt + i + 2 if i < num_images - 1 else -1,
                              'video_id': video_cnt}
                out['images'].append(image_info)
            print('{}: {} images'.format(seq, num_images))

            for i in range(anns.shape[0]):
                frame_id = int(anns[i][0]) - _min + 1
                track_id = int(anns[i][1])
                ann_cnt += 1
                category_id = 1
                ann = {'id': ann_cnt,
                       'category_id': category_id,
                       'image_id': image_cnt + frame_id,
                       'track_id': track_id,
                       'bbox': anns[i][2:6].tolist(),
                       'conf': float(anns[i][6])}
                out['annotations'].append(ann)
            image_cnt += num_images

    print('loaded {} for {} images and {} samples'.format(split, len(out['images']), len(out['annotations'])))
    json.dump(out, open(out_path, 'w'))