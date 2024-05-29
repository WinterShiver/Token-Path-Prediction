from torch import Tensor
from torchmetrics import Metric
import logging
from src.modules.doc_utils import cal_f1
import torch
from nltk.translate.bleu_score import sentence_bleu
import json

logger = logging.getLogger('lightning')

class ReadingOrderBaseMetric(Metric): 
    def __init__(self,
                 compute_on_step=False,
                 dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step, compute_on_step=compute_on_step)
        self.add_state("cnt", default=torch.tensor(0), dist_reduce_fx="sum")
    
    def decode(self, label):
        """ input: ['0_2', '2_3', '3_1', '1_0'], output: [1, 2, 0] (from [0, 2, 3, 1, 0])
        """
        idxs = [0]
        for ij in label:
            i, j = ij.split('_')
            i, j = int(i), int(j)
            assert idxs[-1] == i
            idxs.append(j)
        assert idxs[0] == idxs[-1] == 0
        return [i-1 for i in idxs[1:-1]]


class ReadingOrderBLEUMetric(ReadingOrderBaseMetric):
    def __init__(self,
                 compute_on_step=False,
                 dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step, compute_on_step=compute_on_step)
        self.add_state("total_bleu_seg", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("total_bleu_word", default=torch.tensor(0.), dist_reduce_fx="sum")

    def bleu(self, label, prediction):
        '''
        prediction = [1,2,0]
        label = [0,1,2]
        '''
        assert len(prediction) == len(label)
        # 计算单条样本的BLEU
        if len(label) >= 4:
            # 默认使用BLEU-4
            bleu_score = sentence_bleu([label], prediction)
        else:
            # 当序列长度为n(n<4)时，最多只有n-gram，使用BLEU-n
            weights = [1/len(label) for i in range(len(label))]
            bleu_score = sentence_bleu([label], prediction, weights=weights)
        return bleu_score


    def update(self, predictions, labels, segment_word_ids=None):
        '''
        predictions = [['0_2', '2_3', '3_1', '1_0'], ...]
        labels = [['0_1', '1_2', '2_3', '3_0'], ...]
        '''

        if segment_word_ids is None:
            segment_word_ids = [None for i in predictions]
        
        for prediction, label, id_map in zip(predictions, labels, segment_word_ids):

            # 获得形如segment id list的排序结果（predictions）和实际标签（labels），执行segment-level的metric计算
            prediction, label = self.decode(prediction), self.decode(label)
            self.total_bleu_seg += self.bleu(label, prediction)
            # 如果有segment id to word-id的映射（segment_word_ids）
            # 则根据segment_word_ids还原word order，获得word id list，执行word-level的metric计算
            if not id_map is None:
                id_map = json.loads(id_map)
                prediction = [word_id for segment_id in prediction for word_id in id_map[str(segment_id)]]
                label = [word_id for segment_id in label for word_id in id_map[str(segment_id)]]
                self.total_bleu_word += self.bleu(label, prediction)
            
            self.cnt += 1

    def compute(self):
        metric = {
            "avg_bleu_seg": 100 * self.total_bleu_seg / self.cnt,
            "avg_bleu_word": 100 * self.total_bleu_word / self.cnt,
            "samples": self.cnt,}
        return metric


class ReadingOrderAvgRelDistMetric(ReadingOrderBaseMetric):
    def __init__(self,
                 compute_on_step=False,
                 dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step, compute_on_step=compute_on_step)
        self.add_state("total_dist_seg", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("total_dist_word", default=torch.tensor(0.), dist_reduce_fx="sum")

    def ard(self, prediction, label):
        acc_dist = 0.
        for i, e in enumerate(label):
            try:
                acc_dist += abs(prediction.index(e) - i)
            except Exception as e:
                acc_dist += len(label)
        return acc_dist / len(label)

    def update(self, predictions, labels, segment_word_ids=None):
        '''
        predictions = [['0_2', '2_3', '3_1', '1_0'], ...]
        labels = [['0_1', '1_2', '2_3', '3_0'], ...]
        '''

        if segment_word_ids is None:
            segment_word_ids = [None for i in prediction]

        for prediction, label, id_map in zip(predictions, labels, segment_word_ids):

            # 获得形如segment id list的排序结果（predictions）和实际标签（labels），执行segment-level的metric计算
            prediction, label = self.decode(prediction), self.decode(label)
            self.total_dist_seg += self.ard(prediction, label)

            # 如果有segment id to word-id的映射（segment_word_ids）
            # 则根据segment_word_ids还原word order，获得word id list，执行word-level的metric计算
            if not id_map is None:
                id_map = json.loads(id_map)
                prediction = [word_id for segment_id in prediction for word_id in id_map[str(segment_id)]]
                label = [word_id for segment_id in label for word_id in id_map[str(segment_id)]]
                self.total_dist_word += self.ard(prediction, label)
            
            self.cnt += 1

    def compute(self):
        metric = {
            "avg_dist_seg": self.total_dist_seg / self.cnt,
            "avg_dist_word": self.total_dist_word / self.cnt,
            "samples": self.cnt,}
        return metric

if __name__ == '__main__':

    predictions = [['0_2', '2_3', '3_1', '1_0']]
    labels = [['0_1', '1_2', '2_3', '3_0']]
    m = ReadingOrderBLEUMetric()
    m.update(predictions, labels)
    print(m.compute())
    m = ReadingOrderAvgRelDistMetric()
    m.update(predictions, labels)
    print(m.compute())