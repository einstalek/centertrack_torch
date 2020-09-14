import torch


class Args:
    # Model parameters
    device = 'cuda'
    device_id = 1
    backbone = 'resnet50'
    heads = {'hm': 1, 'reg': 2, 'wh': 2, 'tracking': 2, }
    weights = {'hm': 1.1, 'reg': 1, 'wh': 0.7, 'tracking': 1, }
    use_bias = True
    use_conv_transp = False
    num_kernels = 2
    down_ratio = 8
    num_frames = 1
    inp_dim = 3 + 3 * num_frames

    # Data annotations
    train_json = '/home/jovyan/dataset/train-v2.json'
    val_json = '/home/jovyan/dataset/val-v2.json'
    data_dir = "/home/jovyan/dataset/"

    #     # coco split for batch size 92
    #     train_group_rates = {("hand_dataset", "coco/"): 42,
    #                          ("0002_openpose",): 2,
    #                          ("0002_navigation_",): 6,
    #                          ("0004_",): 6, ("0010_",): 16,
    #                          ("0023_",): 16, ("office",): 2,
    #                          ("human_crop",): 2,
    #                         }

    # split for batch size 92
    train_group_rates = {("hand_dataset", "coco/"): 14,
                         ("0002_openpose",): 2,
                         ("0002_navigation_",): 12,
                         ("0004_",): 12, ("0010_",): 24,
                         ("0023_",): 24, ("office",): 2,
                         ("human_crop",): 2,
                         }

    # Input parameters
    pre_hm = False
    ret_fpath = True
    max_frame_dist = 30
    max_dist_proba = 0.7
    input_w = 384
    input_h = 224
    cvt_gray = False
    widen_boxes = 1.15
    gaussian_min_radius = 2
    pre_hm_min_radius = 16

    # Validation parameters
    val_skip_frame = 2  # frame_dist между frame и prev_frame
    val_select_frame = 5  # frame_num % val_select_frame = 0 для подчета метрик

    # Augmentation parameters
    aug_rot = 1
    rotate = 5
    crop_near_box = 0.2
    crop_min_box_size = 60
    fp_disturb = 0.2
    lost_disturb = 0.4
    hm_disturb = 0.08
    no_color_aug = False
    use_gamma = True
    gamma = (0.3, 2.)
    aug_s = (0.7, 1.1)

    comment = "resnet backbone, non-coco split"

    # Training parameters
    batch_size = 92
    start_epoch = 49
    end_epoch = 400
    write_mota_metrics = True
    num_iters = {'train': 150, 'val': -1}
    gpu = 14
    lr = 1.3e-4
    clip_value = 50.
    lr_step = (start_epoch + 150, start_epoch + 200)
    drop = 0.8
    max_objs = 15
    print_iter = 20
    hm_l1_loss = 0.

    # Checkpoints
    save_dir = "/home/jovyan/CenterTrack/weights_1/"
    res_dir = save_dir + 'temp/'
    weights_dir = save_dir
    load_model = None  # save_dir + "model_48.pth"
    save_point = range(start_epoch + 1, end_epoch + 1, 3)

    # Tracker
    new_thresh = 0.4
    thresh = 0.3

    def __init__(self):
        self.device = torch.device('cuda' if self.device == 'cuda' else 'cpu')
