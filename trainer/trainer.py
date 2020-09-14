import time
import os

import numpy as np

import torch
from trainer.losses import FastFocalLoss, RegWeightedL1Loss, RegBalancedL1Loss, HeatmapFocalLoss
from trainer.utils import AverageMeter
from trainer.utils import generic_decode
from progress.bar import Bar
import motmetrics as mm

from tracker.tracker import Tracker

GT = None


def iou_dist(gt, dr, max_iou=0.5):
    gt_boxes = []
    for box in gt:
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        gt_boxes.append((x1, y1, w, h))
    dr_boxes = []
    for box in dr:
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        dr_boxes.append((x1, y1, w, h))
    return mm.distances.iou_matrix(gt_boxes, dr_boxes, max_iou=max_iou)


def load(fid, gt, dr, use_ids=True, max_iou=0.5):
    gt_boxes, dr_boxes = gt[fid], dr[fid]
    gt_id = np.arange(len(gt_boxes))
    dr_id = dr_boxes['ids']
    dist = iou_dist(gt_boxes, dr_boxes['bboxes'], max_iou=max_iou)
    return gt_id, dr_id, dist


def _sigmoid(x):
    y = torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)
    return y


class GenericLoss(torch.nn.Module):
    def __init__(self, args):
        super(GenericLoss, self).__init__()
        self.crit = FastFocalLoss()
        self.crit_reg = RegWeightedL1Loss()
        self.crit_bal_reg = RegBalancedL1Loss()
        self.crit_heatmap_floss = HeatmapFocalLoss()
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
                losses['hm'] += self.args.hm_l1_loss * self.crit_heatmap_floss(
                    output['hm'], batch['hm'], batch['mask']) / len(weights)
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
        self.preds = {}

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

    def run_epoch(self, phase, epoch, data_loader, rank):
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
        num_iters = len(data_loader) if self.args.num_iters[phase] < 0 else self.args.num_iters[phase]
        bar = Bar('{}'.format("tracking"), max=num_iters)
        end = time.time()
        for iter_id, batch in enumerate(data_loader):
            if iter_id >= num_iters:
                break
            data_time.update(time.time() - end)

            for k in batch:
                if k in ('fpath', 'prev_fpath'):
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
                torch.nn.utils.clip_grad_norm_(model_with_loss.parameters(), self.args.clip_value)
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

            if rank == 0 and phase == 'val' and self.args.write_mota_metrics and epoch in self.args.save_point:
                curr_name = None
                tracker = None
                for i in range(self.args.batch_size):
                    try:
                        fpath = batch['fpath'][i]
                    except IndexError:
                        break
                    fpath = fpath.split('.')[0].split('/')[-1]

                    name, num = fpath.split("_frame_")
                    num = int(num)
                    if num % self.args.val_select_frame != 0:
                        continue

                    if name != curr_name:
                        curr_name = name
                        tracker = Tracker(self.args)

                    out = [x[i][None] for x in output]
                    res = out
                    dets = generic_decode({k: res[t] for (t, k) in enumerate(self.args.heads)},
                                          self.args.max_objs, self.args)
                    for k in dets:
                        dets[k] = dets[k].detach().cpu().numpy()

                    if not tracker.init and len(dets) > 0:
                        tracker.init_track(dets)
                    elif len(dets) > 0:
                        tracker.step(dets)

                    with open(os.path.join(self.args.res_dir, fpath + '.txt'), "w") as f:
                        for track in tracker.tracks:
                            x1, y1, x2, y2 = track['bbox']
                            f.write("{} {} {} {} {} {}\n".format(track['score'],
                                                                 track['tracking_id'],
                                                                 x1, y1, x2, y2))
            if rank == 0 and self.args.print_iter > 0:  # If not using progress bar
                if iter_id % self.args.print_iter == 0:
                    print('{}| {}'.format("tracking", Bar.suffix))
            else:
                bar.next()
            del output, loss, loss_stats

        if rank == 0 and phase == 'val' and self.args.write_mota_metrics and epoch in self.args.save_point:
            self.compute_map(epoch)

        bar.finish()
        ret = {k: v.avg for k, v in avg_loss_stats.items()}
        ret['time'] = bar.elapsed_td.total_seconds() / 60.
        return ret, results

    def train(self, epoch, data_loader, rank):
        return self.run_epoch('train', epoch, data_loader, rank)

    def val(self, epoch, data_loader, rank):
        if rank == 0:
            for fp in os.listdir(self.args.res_dir):
                os.remove(os.path.join(self.args.res_dir, fp))
        return self.run_epoch('val', epoch, data_loader, rank)

    def collect_gt(self):
        global GT
        GT = {}
        root = "/home/jovyan/mAP/input/ground-truth"
        for x in os.listdir(root):
            if int(x.split('.')[0].split('_')[-1]) % 10 != 0:
                continue
            boxes = []
            with open(os.path.join(root, x)) as f:
                for line in f:
                    *bbox, s_h, s_w = line.strip().split()
                    s_h, s_w = float(s_h), float(s_w)
                    x1, y1, x2, y2 = list(map(float, bbox))
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    w, h = self.args.widen_boxes * (x2 - x1), self.args.widen_boxes * (y2 - y1)
                    x1, x2 = cx - w / 2, cx + w / 2
                    y1, y2 = cy - h / 2, cy + h / 2
                    x1, x2 = x1 / s_w, x2 / s_w
                    y1, y2 = y1 / s_h, y2 / s_h
                    boxes.append([x1, y1, x2, y2])
            GT[x.split('.')[0]] = boxes

    def compute_map(self, epoch):
        global GT
        if GT is None:
            self.collect_gt()
        root = self.args.res_dir
        unique_fps = set()
        for x in os.listdir(root):
            name = x.split('.')[0]
            unique_fps.add(name)

        dr = {}
        for x in unique_fps:
            if int(x.split('_')[-1]) % 10 != 0:
                continue
            if x not in dr:
                dr[x] = {'bboxes': [], 'scores': [], 'ids': []}

            with open(os.path.join(root, x + '.txt')) as f:
                tracks = f.read().split('\n')
                tracks = [list(map(float, x.split())) for x in tracks if x]

            for track in tracks:
                score, _id, *box = track
                dr[x]['scores'].append(score)
                dr[x]['bboxes'].append(self.args.down_ratio * np.array(box))
                dr[x]['ids'].append(int(_id))

        inters = set(GT.keys()).intersection(dr.keys())
        unique_fids = set()
        for x in inters:
            unique_fids.add(x.split('.')[0].split('_frame')[0])
        unique_fids = list(unique_fids)
        accs = []
        for k in unique_fids:
            acc = mm.MOTAccumulator(auto_id=True)
            fids = sorted([x.split('.')[0] for x in inters if k in x],
                          key=lambda x: int(x.split('.')[0].split('_')[-1]))
            for fid in fids:
                gt_id, dr_id, dist = load(fid, GT, dr, use_ids=True, max_iou=0.55)
                acc.update(gt_id, dr_id, dist)
            accs.append(acc)
        mh = mm.metrics.create()
        summary = mh.compute_many(
            accs,
            metrics=['num_frames', 'num_objects',
                     'num_predictions', 'num_matches', 'num_misses', 'num_false_positives', 'num_switches',
                     'mota', 'motp', 'recall',
                     'idp', 'idr', 'idf1'
                     ],
            generate_overall=True,
            names=unique_fids,
        )
        summary.to_csv(os.path.join(self.args.weights_dir, "map_result_{}.csv".format(epoch)))
