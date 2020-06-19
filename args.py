import torch


class Args:
    heads = {'hm': 1, 'reg': 2, 'wh': 2, 'tracking': 2, }
    weights = {'hm': 1, 'reg': 1, 'wh': 0.6, 'tracking': 1, }
    train_json = '/home/jovyan/dataset/train.json'
    val_json = '/home/jovyan/dataset/val.json'
    data_dir = "/home/jovyan/dataset/"
    max_frame_dist = 100
    input_w = 384
    input_h = 224
    down_ratio = 4
    fp_disturb = 0.1
    lost_disturb = 0.4
    hm_disturb = 0.05
    no_color_aug = False
    device = 'cuda'
    save_dir = "/home/jovyan/CenterTrack/weights_rect/"
    weights_dir = "/home/jovyan/CenterTrack/weights_rect/"
    batch_size = 56
    start_epoch = 0
    end_epoch = 300
    save_point = range(start_epoch + 1, end_epoch + 1, 2)
    num_iters = 150
    gpu = 14
    lr = 1.3e-5
    lr_step = (start_epoch + 140, start_epoch + 200)
    drop = 0.9
    load_model = "/home/jovyan/CenterTrack/weights/model_189.pth"
    max_objs = 10
    print_iter = 10
    pre_hm = False

    def __init__(self):
        self.device = torch.device('cuda' if self.device=='cuda' else 'cpu')

