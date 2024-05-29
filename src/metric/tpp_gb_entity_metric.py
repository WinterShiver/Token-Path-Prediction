from torch import Tensor
from torchmetrics import Metric
import logging
from src.modules.doc_utils import cal_f1
import torch

logger = logging.getLogger('lightning')


class TPPGB2NERMetric(Metric):
    def __init__(self,
                 compute_on_step=False,
                 dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step, compute_on_step=compute_on_step)
        self.add_state("p", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("r", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("c", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("cnt", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, predictions, labels):
        '''
        predictions = [['0:1-2-3','1:4','2:7-8-9'],['0:4-5']]
        labels = [['0:1-2-3','1:5'],['2:4-5']]
        '''
        for prediction, label in zip(predictions, labels):
            p_set = set(prediction)
            t_set = set(label)
            self.p += len(p_set)
            self.r += len(t_set)
            self.c += len(p_set & t_set)
            self.cnt += 1

    def compute(self):
        e_f1, e_p, e_r = cal_f1(self.c, self.p, self.r)
        metric = {"f1": e_f1,
                  "precision": e_p,
                  "recall": e_r,
                  "samples": self.cnt,
                  }

        return metric
