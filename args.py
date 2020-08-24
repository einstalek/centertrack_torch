import torch


class Args:
    device = 'cuda'
    heads = {'hm': 1, 'reg': 2, 'wh': 2, 'tracking': 2, }
    weights = {'hm': 1.1, 'reg': 1, 'wh': 0.7, 'tracking': 1, }
    use_bias = True
    use_conv_transp = False
    num_kernels = 2
    down_ratio = 8
    num_frames = 1
    inp_dim = 3 + 3 * num_frames

    train_json = '/home/jovyan/dataset/train.json'
    val_json = '/home/jovyan/dataset/val.json'
    data_dir = "/home/jovyan/dataset/"

    ret_fpath = True
    max_frame_dist = 40
    max_dist_proba = 0.8
    input_w = 384
    input_h = 224
    aug_rot = 1
    rotate = 2
    fp_disturb = 0.2
    lost_disturb = 0.4
    hm_disturb = 0.08
    no_color_aug = False
    cvt_gray = False
    aug_s = (0.75, 1.2)
    widen_boxes = 1.15
    gaussian_min_radius = 2
    pre_hm_min_radius = 16
    save_dir = "/home/jovyan/CenterTrack/weights/"
    weights_dir = "/home/jovyan/CenterTrack/weights/"
    batch_size = 192
    start_epoch = 0
    end_epoch = 500
    save_point = range(start_epoch + 1, end_epoch + 1, 3)
    write_mota_metrics = True
    num_iters = {'train': 200, 'val': -1}
    gpu = 14
    lr = 1e-4
    clip_value = 50.
    lr_step = (start_epoch + 150, start_epoch + 200)
    drop = 0.8
    load_model = '/home/jovyan/CenterTrack/weights/init.pth'
    max_objs = 15
    print_iter = 20
    pre_hm = False
    res_dir = '/home/jovyan/CenterTrack/weights/temp/'

    new_thresh = 0.4
    thresh = 0.3

    comment = "augmentations, more smart crops, init from model_model_136"

    def __init__(self):
        self.device = torch.device('cuda' if self.device == 'cuda' else 'cpu')

