import torch
import torch.nn.functional as F
from torch import nn

from trainer.utils import _only_neg_loss, _tranpose_and_gather_feat


class FastFocalLoss(nn.Module):
    """
    Reimplemented focal loss, exactly the same as the CornerNet version.
    Faster and costs much less memory.
    """

    def __init__(self):
        super(FastFocalLoss, self).__init__()
        self.only_neg_loss = _only_neg_loss

    def forward(self, out, target, ind, mask, cat):
        """
        Arguments:
        out, target: B x C x H x W
        ind, mask: B x M
        cat (category id for peaks): B x M
        """
        neg_loss = self.only_neg_loss(out, target)
        pos_pred_pix = _tranpose_and_gather_feat(out, ind)  # B x M x C
        pos_pred = pos_pred_pix.gather(2, cat.unsqueeze(2))  # B x M
        num_pos = mask.sum()
        pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2) * \
                   mask.unsqueeze(2)
        pos_loss = pos_loss.sum()
        if num_pos == 0:
            return - neg_loss
        return - (pos_loss + neg_loss) / num_pos


class HeatmapFocalLoss(nn.Module):
    def __init__(self):
        super(HeatmapFocalLoss, self).__init__()
        self.only_neg_loss = _only_neg_loss

    def forward(self, out, target, mask):
        pos_loss = F.l1_loss(out, target, reduction='sum')
        return pos_loss / mask.sum()


class RegWeightedL1Loss(nn.Module):
    def __init__(self):
        super(RegWeightedL1Loss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _tranpose_and_gather_feat(output, ind)
        # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
        loss = F.l1_loss(pred * mask, target * mask, reduction='sum')
        loss = loss / (mask.sum() + 1e-4)
        return loss


class RegBalancedL1Loss(nn.Module):
    def __init__(self):
        super(RegBalancedL1Loss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _tranpose_and_gather_feat(output, ind)
        diff = pred * mask - target * mask
        weights = (diff < 0).float()
        weights += 4.
        weights /= 5.  # 0.8 for positive diff and 1.0 for negative
        loss = torch.sum(weights * torch.abs(diff))
        loss = loss / (mask.sum() + 1e-4)
        return loss
