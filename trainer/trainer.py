import time

import torch
from trainer.losses import FastFocalLoss, RegWeightedL1Loss, RegBalancedL1Loss
from trainer.utils import AverageMeter
from progress.bar import Bar


def _sigmoid(x):
    y = torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)
    return y


class GenericLoss(torch.nn.Module):
    def __init__(self, args):
        super(GenericLoss, self).__init__()
        self.crit = FastFocalLoss()
        self.crit_reg = RegWeightedL1Loss()
        self.crit_bal_reg = RegBalancedL1Loss()
        self.args = args
        self.heads = args.heads

    @staticmethod
    def _sigmoid_output(output):
        if 'hm' in output:
            output['hm'] = _sigmoid(output['hm'])
        return output

    def forward(self, outputs, batch):
        weights = self.args.weights
        losses = {k: 0 for k in weights}
        outputs = [{k: outputs[i]} for (i, k) in enumerate(weights.keys())]
        for s in range(len(outputs)):
            output = self._sigmoid_output(outputs[s])

            if 'hm' in output:
                losses['hm'] += self.crit(
                    output['hm'], batch['hm'], batch['ind'],
                    batch['mask'], batch['cat']) / len(weights)

            regression_heads = [
                'reg', 'tracking', 'ltrb', 'ltrb_amodal', 'hps',
                'dep', 'dim']
            for head in regression_heads:
                if head in output:
                    losses[head] += self.crit_reg(
                        output[head], batch[head + '_mask'],
                        batch['ind'], batch[head]) / len(weights)

            bal_regression_heads = ['wh', ]
            for head in bal_regression_heads:
                if head in output:
                    losses[head] += self.crit_bal_reg(
                        output[head], batch[head + '_mask'],
                        batch['ind'], batch[head]) / len(weights)

        losses['tot'] = 0
        for head in self.heads:
            losses['tot'] += weights[head] * losses[head]
        return losses['tot'], losses


class ModelWithLoss(torch.nn.Module):
    def __init__(self, model, loss, args):
        super(ModelWithLoss, self).__init__()
        self.model = model
        self.loss = loss
        self.args = args

    def forward(self, batch):
        pre_img = batch['pre_img'] if 'pre_img' in batch else None
        pre_hm = batch['pre_hm'] if 'pre_hm' in batch else None
        image = batch['image']
        if len(batch['prev_frames']) > 0:
            pre_img = torch.cat([pre_img, *batch['prev_frames']], axis=1)
        if self.args.pre_hm:
            outputs = self.model(torch.cat([image, pre_img, pre_hm], axis=1))
        else:
            outputs = self.model(torch.cat([image, pre_img], axis=1))
        loss, loss_stats = self.loss(outputs, batch)
        return outputs, loss, loss_stats


class Trainer(object):
    def __init__(self, model, optimizer, args):
        self.heads = args.heads
        self.args = args
        self.loss_stats, self.loss = self._get_losses()
        self.model_with_loss = ModelWithLoss(model, self.loss, args)
        self.optimizer = optimizer

    def _get_losses(self):
        heads = self.heads
        loss_order = ['hm', 'wh', 'reg', 'ltrb', 'hps', 'hm_hp',
                      'hp_offset', 'dep', 'dim', 'rot', 'amodel_offset',
                      'ltrb_amodal', 'tracking', 'nuscenes_att', 'velocity']
        loss_states = ['tot'] + [k for k in loss_order if k in heads]
        loss = GenericLoss(self.args)
        return loss_states, loss

    def set_device(self, device):
        self.model_with_loss = self.model_with_loss.to(device)

        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device=device, non_blocking=True)

    def run_epoch(self, phase, epoch, data_loader, rank=1):
        model_with_loss = self.model_with_loss
        if phase == 'train':
            model_with_loss.train()
        else:
            model_with_loss.eval()
            torch.cuda.empty_cache()

        results = {}
        data_time, batch_time = AverageMeter(), AverageMeter()
        avg_loss_stats = {l: AverageMeter() for l in self.loss_stats
                          if l in ('tot', 'hm', 'wh', 'tracking')}
        num_iters = len(data_loader)
        bar = Bar('{}'.format("tracking"), max=num_iters)
        end = time.time()

        num_iters = len(data_loader) if self.args.num_iters < 0 else self.args.num_iters
        for iter_id, batch in enumerate(data_loader):
            if iter_id >= num_iters:
                break
            data_time.update(time.time() - end)

            for k in batch:
                if k == 'meta':
                    continue
                if type(batch[k]) != list:
                    batch[k] = batch[k].to(self.args.device, non_blocking=True)
                else:
                    for i in range(len(batch[k])):
                        batch[k][i] = batch[k][i].to(self.args.device, non_blocking=True)

            output, loss, loss_stats = model_with_loss(batch)
            loss = loss.mean()
            if phase == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model_with_loss.parameters(), 10.)
                self.optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            Bar.suffix = '{phase}: [{0}][{1}/{2}]| '.format(
                epoch, iter_id, num_iters, phase=phase)
            for l in avg_loss_stats:
                avg_loss_stats[l].update(
                    loss_stats[l].mean().item(), batch['image'].size(0))
                Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)
            Bar.suffix = Bar.suffix + '|Data {dt.val:.3f}s({dt.avg:.3f}s) ' \
                                      '|Net {bt.avg:.3f}s'.format(dt=data_time, bt=batch_time)

            if rank == 0 and self.args.print_iter > 0:  # If not using progress bar
                if iter_id % self.args.print_iter == 0:
                    print('{}| {}'.format("tracking", Bar.suffix))
            else:
                bar.next()

            del output, loss, loss_stats

        bar.finish()
        ret = {k: v.avg for k, v in avg_loss_stats.items()}
        ret['time'] = bar.elapsed_td.total_seconds() / 60.
        return ret, results

    def train(self, epoch, data_loader, rank=1):
        return self.run_epoch('train', epoch, data_loader, rank)

    def val(self, epoch, data_loader, rank=1):
        return self.run_epoch('val', epoch, data_loader, rank)
