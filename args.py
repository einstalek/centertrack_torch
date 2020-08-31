import torch


class Args:
    # Model parameters
    device = 'cuda'
    backbone = 'blazepalm'
    heads = {'hm': 1, 'reg': 2, 'wh': 2, 'tracking': 2, }
    weights = {'hm': 1.1, 'reg': 1, 'wh': 0.7, 'tracking': 1, }
    use_bias = True
    use_conv_transp = False
    num_kernels = 2
    down_ratio = 8
    num_frames = 1
    inp_dim = 3 + 3 * num_frames

    # Data annotations
    train_json = '/home/jovyan/dataset/train.json'
    val_json = '/home/jovyan/dataset/val.json'
    data_dir = "/home/jovyan/dataset/"

    #     # split for batch size 192
    #     train_group_rates = {("hand_dataset", "coco/"): 8,
    #                          ("0002_openpose/",): 8,
    #                          ("0002_navigation_",): 48,
    #                          ("0004_",): 44, ("0010_",): 48,
    #                          ("0018_",): 8, ("office",): 28,
    #                         }
    #     val_group_rates = {("0002_navigation_",): 48,
    #                        ("0004_",): 48, ("0010_",): 48,
    #                        ("office",): 48,
    #                        ("hand_dataset", "coco/"): 0
    #                       }

    # split for batch size 64
    train_group_rates = {("hand_dataset", "coco/"): 8,
                         ("0002_openpose/",): 6,
                         ("0002_navigation_",): 12,
                         ("0004_",): 12, ("0010_",): 12,
                         ("0018_",): 6, ("office",): 8,
                         }
    val_group_rates = {("0002_navigation_",): 16,
                       ("0004_",): 16, ("0010_",): 16,
                       ("office",): 16,
                       ("hand_dataset", "coco/"): 0
                       }

    # Input parameters
    pre_hm = False
    ret_fpath = True
    max_frame_dist = 30
    max_dist_proba = 0.8
    input_w = 384
    input_h = 224
    cvt_gray = False
    widen_boxes = 1.15
    gaussian_min_radius = 2
    pre_hm_min_radius = 16

    # Validation parameters
    val_skip_frame = 1  # frame_dist между frame и prev_frame
    val_select_frame = 10  # frame_num % val_select_frame = 0 для подчета метрик

    # Augmentation parameters
    aug_rot = 1
    rotate = 5
    fp_disturb = 0.2
    lost_disturb = 0.4
    hm_disturb = 0.08
    no_color_aug = False
    use_gamma = True
    gamma = (0.3, 2.)
    aug_s = (0.75, 1.2)

    comment = "mobilenet backbone"

    # Training parameters
    batch_size = 64
    start_epoch = 0
    end_epoch = 300
    write_mota_metrics = True
    num_iters = {'train': 100, 'val': -1}
    gpu = 14
    lr = 1e-3
    clip_value = 50.
    lr_step = (start_epoch + 150, start_epoch + 200)
    drop = 0.8
    max_objs = 15
    print_iter = 20

    # Checkpoints
    save_dir = "/home/jovyan/CenterTrack/weights_2/"
    res_dir = save_dir + 'temp/'
    weights_dir = save_dir
    load_model = None
    save_point = range(start_epoch + 1, end_epoch + 1, 3)

    # Tracker
    new_thresh = 0.4
    thresh = 0.3

    def __init__(self):
        self.device = torch.device('cuda' if self.device == 'cuda' else 'cpu')

