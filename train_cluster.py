import horovod.torch as hvd
hvd.init()

import os
import sys
sys.path.insert(0, '/home/jovyan/env/lib/python3.6/site-packages/')

import torch
torch.cuda.set_device(hvd.local_rank())

import torch.utils.data

from args import Args

args = Args()
f = open(os.path.join(args.save_dir, "std.txt"), 'w', buffering=1, encoding='utf-8')
sys.stderr=f
sys.stdout=f

from blazepalm.detector import BlazePalm, save_model
from dataset.generic_dataset import GenericDataset
from trainer.logger import Logger
from trainer.trainer import Trainer


if __name__ == '__main__':
    if hvd.rank() == 0:
        logger = Logger(args)
    blaze_palm = BlazePalm(args)
    opt = torch.optim.Adam(blaze_palm.parameters(), lr=args.lr * hvd.size())
    
    if args.load_model:
        state_dict = torch.load(args.load_model)
        blaze_palm.load_state_dict(state_dict['state_dict'])
    
    
    opt = hvd.DistributedOptimizer(opt, named_parameters=blaze_palm.named_parameters())
    
    trainer = Trainer(blaze_palm, opt, args)
    trainer.set_device(args.device)
    data = GenericDataset(args,
                          args.train_json,
                          args.data_dir, 
                          split='train')
    train_sampler = torch.utils.data.distributed.DistributedSampler(data, 
                                                                    num_replicas=hvd.size(), 
                                                                    rank=hvd.rank())
    train_loader = torch.utils.data.DataLoader(
        data, batch_size=args.batch_size, num_workers = args.num_workers,
        pin_memory=True, drop_last=True, sampler=train_sampler,
    )
    
    val_data = GenericDataset(args,
                              args.val_json,
                              args.data_dir,
                              split='val')
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_data, 
                                                                  num_replicas=hvd.size(), 
                                                                  rank=hvd.rank())
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=args.batch_size, num_workers = args.num_workers,
        pin_memory=True, drop_last=True, sampler=val_sampler,
    )
    
    hvd.broadcast_parameters(blaze_palm.state_dict(), root_rank=0)

    written = False
    
    for epoch in range(1 + args.start_epoch, 1 + args.end_epoch):
        mark = epoch
        log_dict_train, _ = trainer.train(epoch, train_loader, rank=hvd.rank())
        
        if epoch in args.lr_step:
            lr = args.lr * (args.drop ** (args.lr_step.index(epoch) + 1))
            print('Drop LR to', lr)
            for param_group in opt.param_groups:
                param_group['lr'] = lr
        
        if not written:
            os.system("nvidia-smi")
            written = True
        
        if hvd.rank() == 0:
            logger.write('epoch: {} |'.format(epoch))
            for k, v in log_dict_train.items():
                logger.write('{} {:8f} | '.format(k, v))
            logger.write('\n')
            
            with torch.no_grad():
                log_dict_val, preds = trainer.val(epoch, val_loader)
                for k, v in log_dict_val.items():
                    logger.write('{} {:8f} | '.format(k, v))
                logger.write('\n')
                
            if epoch in args.save_point:
                save_model(os.path.join(args.weights_dir, 'model_{}.pth'.format(epoch)),
                           epoch, blaze_palm, opt)
            






