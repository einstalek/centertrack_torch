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
    max_dist_proba = 0.6
    input_w = 384
    input_h = 224
    fp_disturb = 0.1
    lost_disturb = 0.4
    hm_disturb = 0.05
    no_color_aug = False
    aug_s = (0.6, 1.1)
    save_dir = "/home/jovyan/CenterTrack/weights_5/"
    weights_dir = "/home/jovyan/CenterTrack/weights_5/"
    batch_size = 184
    start_epoch = 0
    end_epoch = 300
    save_point = range(start_epoch + 1, end_epoch + 1, 3)
    write_mota_metrics = True
    num_iters = {'train': 222, 'val': 60}
    gpu = 14
    lr = 1.3e-5
    clip_value = 50.
    lr_step = (start_epoch + 220, start_epoch + 250)
    drop = 0.95
    load_model = "/home/jovyan/CenterTrack/weights_5/init.pth"
    max_objs = 10
    print_iter = 10
    pre_hm = False
    res_dir = '/home/jovyan/CenterTrack/weights_5/temp/'

    def __init__(self):
        self.device = torch.device('cuda' if self.device == 'cuda' else 'cpu')

