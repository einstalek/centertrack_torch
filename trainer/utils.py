import torch
from torch import nn


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count


def _only_neg_loss(pred, gt):
    gt = torch.pow(1 - gt, 4)
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * gt
    return neg_loss.sum()


def _gather_feat(feat, ind):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    return feat


def _tranpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2
    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def _topk(scores, K=100):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()
    topk_inds = _gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def generic_decode(output, K=100, opt=None):
    if not ('hm' in output):
        return {}

    heat = output['hm']
    batch, cat, height, width = heat.size()

    heat = _nms(heat)
    scores, inds, clses, ys0, xs0 = _topk(heat, K=K)

    clses = clses.view(batch, K)
    scores = scores.view(batch, K)
    bboxes = None
    cts = torch.cat([xs0.unsqueeze(2), ys0.unsqueeze(2)], dim=2)
    ret = {'scores': scores, 'clses': clses.float(),
           'xs': xs0, 'ys': ys0, 'cts': cts}
    if 'reg' in output:
        reg = output['reg']
        reg = _tranpose_and_gather_feat(reg, inds)
        reg = reg.view(batch, K, 2)
        xs = xs0.view(batch, K, 1) + reg[:, :, 0:1]
        ys = ys0.view(batch, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs0.view(batch, K, 1) + 0.5
        ys = ys0.view(batch, K, 1) + 0.5

    if 'wh' in output:
        wh = output['wh']
        wh = _tranpose_and_gather_feat(wh, inds)  # B x K x (F)
        # wh = wh.view(batch, K, -1)
        wh = wh.view(batch, K, 2)
        wh[wh < 0] = 0
        if wh.size(2) == 2 * cat:  # cat spec
            wh = wh.view(batch, K, -1, 2)
            cats = clses.view(batch, K, 1, 1).expand(batch, K, 1, 2)
            wh = wh.gather(2, cats.long()).squeeze(2)  # B x K x 2
        else:
            pass
        bboxes = torch.cat([xs - wh[..., 0:1] / 2,
                            ys - wh[..., 1:2] / 2,
                            xs + wh[..., 0:1] / 2,
                            ys + wh[..., 1:2] / 2], dim=2)
        ret['bboxes'] = bboxes

    regression_heads = ['tracking', 'dep', 'rot', 'dim', 'amodel_offset',
                        'nuscenes_att', 'velocity']

    for head in regression_heads:
        if head in output:
            ret[head] = _tranpose_and_gather_feat(
                output[head], inds).view(batch, K, -1)

    if 'ltrb_amodal' in output:
        ltrb_amodal = output['ltrb_amodal']
        ltrb_amodal = _tranpose_and_gather_feat(ltrb_amodal, inds)  # B x K x 4
        ltrb_amodal = ltrb_amodal.view(batch, K, 4)
        bboxes_amodal = torch.cat([xs0.view(batch, K, 1) + ltrb_amodal[..., 0:1],
                                   ys0.view(batch, K, 1) + ltrb_amodal[..., 1:2],
                                   xs0.view(batch, K, 1) + ltrb_amodal[..., 2:3],
                                   ys0.view(batch, K, 1) + ltrb_amodal[..., 3:4]], dim=2)
        ret['bboxes_amodal'] = bboxes_amodal
        ret['bboxes'] = bboxes_amodal

    return ret
