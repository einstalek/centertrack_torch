from args import Args
import sys
args = Args()

import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)

f = open(os.path.join(args.save_dir, "std.txt"), 'w', buffering=1, encoding='utf-8')
sys.stderr=f
sys.stdout=f

import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import torch
import torch.utils.data

from blazepalm.detector import BlazePalm, save_model
from dataset.generic_dataset import GenericDataset
from trainer.logger import Logger
from trainer.trainer import Trainer

blaze_palm = BlazePalm(args)
if args.load_model:
    state_dict = torch.load(args.load_model)
    blaze_palm.load_state_dict(state_dict['state_dict'])

opt = torch.optim.Adam(blaze_palm.parameters(), lr=args.lr)
logger = Logger(args)

trainer = Trainer(blaze_palm, opt, args)
trainer.set_device(args.device)

data = GenericDataset(args,
                      args.train_json,
                      args.data_dir,
                      split='train',
                      group_rates=args.train_group_rates)
# train_sampler = torch.utils.data.distributed.DistributedSampler(data, num_replicas=1, rank=0)
train_loader = torch.utils.data.DataLoader(
    data, batch_size=args.batch_size,
    drop_last=True, sampler=None,
    num_workers=8,
    timeout=30,
)

val_data = GenericDataset(args,
                          args.val_json,
                          args.data_dir,
                          split='val',
                          group_rates=None)
# val_sampler = torch.utils.data.distributed.DistributedSampler(val_data, num_replicas=1, rank=0)
val_loader = torch.utils.data.DataLoader(
    val_data, batch_size=args.batch_size,
    drop_last=False, sampler=None,
    num_workers=8,
    timeout=30,
)

for epoch in range(1 + args.start_epoch, 1 + args.end_epoch):
    done = False
    while not done:
        try:
            log_dict_train, _ = trainer.train(epoch, train_loader, rank=0)
            done = True
        except RuntimeError:
            del train_loader
            train_loader = torch.utils.data.DataLoader(
                data, batch_size=args.batch_size,
                drop_last=True, sampler=None,
                num_workers=8,
                timeout=30,
            )
            continue

    if epoch in args.lr_step:
        lr = args.lr * (args.drop ** (args.lr_step.index(epoch) + 1))
        print('Drop LR to', lr)
        for param_group in opt.param_groups:
            param_group['lr'] = lr

    logger.write('epoch: {} |'.format(epoch))
    for k, v in log_dict_train.items():
        logger.write('{} {:8f} | '.format(k, v))
    logger.write('\n')

    with torch.no_grad():
        log_dict_val, preds = trainer.val(epoch, val_loader, rank=0)
        for k, v in log_dict_val.items():
            logger.write('{} {:8f} | '.format(k, v))
        logger.write('\n')

    if epoch in args.save_point:
        save_model(os.path.join(args.weights_dir, 'model_{}.pth'.format(epoch)),
                   epoch, blaze_palm, opt)

