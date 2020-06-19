import os

import torch.utils.data

from args import Args
from blazepalm.detector import BlazePalm, save_model
from dataset.generic_dataset import GenericDataset
from trainer.logger import Logger
from trainer.trainer import Trainer


if __name__ == '__main__':
    args = Args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    logger = Logger(args)
    blaze_palm = BlazePalm(args)
    opt = torch.optim.Adam(blaze_palm.parameters())
    trainer = Trainer(blaze_palm, opt, args)
    trainer.set_device(args.device)

    data = GenericDataset(args,
                          args.train_json,
                          args.data_dir)
    train_loader = torch.utils.data.DataLoader(
        data, batch_size=args.batch_size, shuffle=True,
        pin_memory=True, drop_last=True
    )

    # TODO: remove
    val_loader = train_loader

    for epoch in range(1 + args.start_epoch, 1 + args.end_epoch):
        mark = epoch
        log_dict_train, _ = trainer.train(epoch, train_loader)
        logger.write('epoch: {} |'.format(epoch))
        for k, v in log_dict_train.items():
            logger.write('{} {:8f} | '.format(k, v))
        logger.write('\n')
        if epoch in args.save_point:
            save_model(os.path.join(args.save_dir, 'model_{}.pth'.format(epoch)),
                       epoch, blaze_palm, opt)
            with torch.no_grad():
                log_dict_val, preds = trainer.val(epoch, val_loader)
                for k, v in log_dict_val.items():
                    logger.write('{} {:8f} | '.format(k, v))





