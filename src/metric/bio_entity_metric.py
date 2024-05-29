import json
import torch
from torch import Tensor
from torchmetrics import Metric
from src.metric.my_seqeval import Seqeval
import logging
from datasets import ClassLabel
from src.utils.ner_utils import get_entity_bio
# from src.modules.doc_utils import cal_f1

def cal_f1(c, p, r):
    if r == 0 or p == 0:
        return 0, 0, 0

    r = c / r if r else 0
    p = c / p if p else 0

    if r and p:
        return 2 * p * r / (p + r), p, r
    return 0, p, r

logger = logging.getLogger('lightning')


class BIOEntityMetricV2(Metric):
    '''
    原metric特别慢，这里不保存中间结果，每个step计算指标
    '''

    def __init__(self,
                 ner_labels: ClassLabel,
                 compute_on_step=False,
                 dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step, compute_on_step=compute_on_step)

        self.add_state("p", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("r", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("c", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("cnt", default=torch.tensor(0), dist_reduce_fx="sum")
        self.ner_labels = ner_labels

    def update(self, predictions, labels, labels_disc=None):
        '''
                这里需要转为BIO Label
                self.ner_labels.int2str(int(p))
                self.ner_labels.int2str(int(l))
                labels_disc: 形如["answer-34-39-54-59-69-69", "question-175-177"]（字符串形式），为discontinuous时的gold标签
        '''
        bio_predictions = [
            [self.ner_labels.int2str(int(p)) for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        bio_labels = [
            [self.ner_labels.int2str(int(l)) for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        # for bio_prediction, bio_label, label in zip(bio_predictions, bio_labels, labels_disc):
        #     true_labels = ['-'.join([str(v) for v in vs]) for vs in get_entity_bio(bio_label)]
        #     print(true_labels)
        #     true_labels = json.loads(label)
        #     print(true_labels)
        #     input()
        if labels_disc is None:
            for bio_prediction, bio_label in zip(bio_predictions, bio_labels):
                pred_labels = ['-'.join([str(v) for v in vs]) for vs in get_entity_bio(bio_prediction)]
                true_labels = ['-'.join([str(v) for v in vs]) for vs in get_entity_bio(bio_label)]
                p_set = set(pred_labels)
                t_set = set(true_labels)
                self.p += len(p_set)
                self.r += len(t_set)
                self.c += len(p_set & t_set)
                self.cnt += 1
        else:
            for bio_prediction, label in zip(bio_predictions, labels_disc):
                pred_labels = ['-'.join([str(v) for v in vs]) for vs in get_entity_bio(bio_prediction)]
                true_labels = json.loads(label)
                p_set = set(pred_labels)
                t_set = set(true_labels)
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


class BIOEntityMetric(Metric):
    def __init__(self,
                 ner_labels: ClassLabel,
                 compute_on_step=False,
                 dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step, compute_on_step=compute_on_step)

        self.add_state("pred", default=[], dist_reduce_fx="cat")
        self.add_state("true", default=[], dist_reduce_fx="cat")
        self.add_state("samples", default=[], dist_reduce_fx="cat")

        self.ner_labels = ner_labels

    def update(self, predictions, labels):
        '''
        这里需要转为BIO Label
        self.ner_labels.int2str(int(p))
        self.ner_labels.int2str(int(l))
        '''
        bio_predictions = [
            [p for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        bio_labels = [
            [l for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]


        for bio_prediction in bio_predictions:
            self.pred.extend(bio_prediction)
        for bio_label in bio_labels:
            self.true.extend(bio_label)

        self.samples.extend([1] * len(predictions))

    def compute(self):

        predictions = [self.ner_labels.int2str(int(p)) for p in self.pred]
        labels = [self.ner_labels.int2str(int(l)) for l in self.true]
        seqeval = Seqeval()
        results = seqeval.compute(predictions=[predictions], references=[labels])
        metric = {"micro_f1": results['overall_f1'],
                  "precision": results['overall_precision'],
                  "recall": results['overall_recall'],
                  "accuracy": results['overall_accuracy'],
                  'samples': len(self.samples)
                  }

        return metric

    def __hash__(self) -> int:
        # we need to add the id here, since PyTorch requires a module hash to be unique.
        # Internally, PyTorch nn.Module relies on that for children discovery
        # (see https://github.com/pytorch/pytorch/blob/v1.9.0/torch/nn/modules/module.py#L1544)
        # For metrics that include tensors it is not a problem,
        # since their hash is unique based on the memory location but we cannot rely on that for every metric.
        hash_vals = [self.__class__.__name__, id(self)]

        for key in self._defaults:
            val = getattr(self, key)
            # Special case: allow list values, so long
            # as their elements are hashable
            if hasattr(val, "__iter__") and not isinstance(val, Tensor):
                hash_vals.extend(val)
            else:
                hash_vals.append(val)

        return hash(tuple(hash_vals))
