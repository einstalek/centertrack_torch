import os
import numpy as np
import json

DATA_PATH = '/Users/einstalek/abc_set/'
OUT_PATH = os.path.join(DATA_PATH, '')
split = "trainer"

if __name__ == '__main__':
    out_path = os.path.join(OUT_PATH, '{}.json'.format(split))
    out = {'images': [], 'annotations': [],
           'categories': [{'id': 1, 'name': 'human'}],
           'videos': []}
    seqs = ['DSC_8685']
    image_cnt = 0
    ann_cnt = 0
    video_cnt = 0
    for seq in seqs:
        if '.DS_Store' in seq:
            continue
        video_cnt += 1
        out['videos'].append({'id': video_cnt, 'file_name': seq})
        seq_path = os.path.join(DATA_PATH, seq)
        img_path = seq_path
        ann_path = os.path.join(seq_path, "{}.txt".format(seq))
        images = os.listdir(img_path)
        num_images = len([x for x in images if '.npy' in x])

        try:
            anns = np.loadtxt(ann_path, dtype=np.float32, delimiter=',')
        except:
            print("No mot annotation for {}".format(seq))
            continue

        for i in range(num_images):
            image_info = {'file_name': '{}/frame_{}.npy'.format(seq, i),
                          'id': image_cnt + i,
                          'frame_id': i,
                          'prev_image_id': image_cnt + i - 1 if i > 0 else -1,
                          'next_image_id': image_cnt + i + 1 if i < num_images - 1 else -1,
                          'video_id': video_cnt}
            out['images'].append(image_info)

        print('{}: {} images'.format(seq, num_images))

        for i in range(anns.shape[0]):
            frame_id = int(anns[i][0])
            track_id = int(anns[i][1])
            #             cat_id = int(anns[i][7])
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

