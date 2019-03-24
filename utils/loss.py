"""Custom losses."""
import torch
import torch.nn as nn


class MixSoftmaxCrossEntropyLoss(nn.Module):
    def __init__(self, aux=True, aux_weight=0.2, ignore_label=255, **kwargs):
        super(MixSoftmaxCrossEntropyLoss, self).__init__()
        self.aux = aux
        self.aux_weight = aux_weight
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_label)

    def _aux_forward(self, pred1, pred2, label, **kwargs):
        loss1 = self.criterion(pred1, label, **kwargs)
        loss2 = self.criterion(pred2, label, **kwargs)

        return loss1 + self.aux_weight * loss2

    def forward(self, pred, label, **kwargs):
        if self.aux:
            assert (len(pred) == 2)
            return self._aux_forward(pred[0], pred[1], label, **kwargs)
        else:
            assert (len(pred) == 1)
            return self.criterion(pred[0], label, **kwargs)
