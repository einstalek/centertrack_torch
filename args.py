import torch


class Args:
    device = 'cuda'
    heads = {'hm': 1, 'reg': 2, 'wh': 2, 'tracking': 2, }
    weights = {'hm': 1, 'reg': 1, 'wh': 0.6, 'tracking': 1, }
    use_bias = True
    use_conv_transp = False
    num_kernels = 2
    down_ratio = 8
    num_frames = 1
    inp_dim = 3 + 3 * num_frames

    train_json = '/home/jovyan/dataset/train.json'
    val_json = '/home/jovyan/dataset/val.json'
    data_dir = "/home/jovyan/dataset/"

    max_frame_dist = 120
    input_w = 384
    input_h = 224
    fp_disturb = 0.1
    lost_disturb = 0.4
    hm_disturb = 0.05
    no_color_aug = False
    aug_s = (0.5, 1.1)
    save_dir = "/home/jovyan/CenterTrack/weights_2/"
    weights_dir = "/home/jovyan/CenterTrack/weights_2/"
    batch_size = 56
    start_epoch = 299
    end_epoch = 500
    save_point = range(start_epoch + 1, end_epoch + 1, 3)
    num_iters = 300
    gpu = 14
    lr = 1e-5
    lr_step = (start_epoch + 150, start_epoch + 200)
    drop = 0.85
    load_model = "/home/jovyan/CenterTrack/weights_2/model_299.pth"
    max_objs = 10
    print_iter = 50
    pre_hm = False
    num_workers = 8

    def __init__(self):
        self.device = torch.device('cuda' if self.device == 'cuda' else 'cpu')

