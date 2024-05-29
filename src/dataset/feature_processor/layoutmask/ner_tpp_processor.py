
import json
from src.dataset.feature_processor.base_processor import CodeDocBaseProcessor
from src.modules.doc_utils import *


class CodeDocForTPP2NerProcessor(CodeDocBaseProcessor):
    '''
    CodeDocForTPP2NerProcessor
    '''

    def __init__(self,
                 tokenizer,

                 ner_labels,

                 max_text_length=512,

                 box_level="segment",  # or word

                 norm_bbox_width=1000,
                 norm_bbox_height=1000,

                 use_global_1d=False,
                 augment_processors=None,
                 **kwargs,
                 ):

        super().__init__(tokenizer=tokenizer,
                         max_text_length=max_text_length,
                         box_level=box_level,
                         norm_bbox_height=norm_bbox_height,
                         norm_bbox_width=norm_bbox_width,
                         use_global_1d=use_global_1d,
                         augment_processors=augment_processors,
                         **kwargs)
        self.ner_labels = ner_labels

    def process_label(self, sample, word2token_idx):
        '''
        生成下游任务Label
        使用稀疏矩阵表示
        '''
        result = dict()
        json_obj = sample['json']

        # TPP2NER label #
        # _grid_labels = np.zeros((self.ner_labels.num_classes, self.max_text_length, self.max_text_length), dtype=np.int)

        _grid_indexs = [[], [], []]
        _grid_values = []

        _entity_text = set()

        label_entities = json_obj.get('label_entities', [])
        for label_entity in label_entities:

            label = label_entity['label']
            label_id = self.ner_labels.str2int(label)

            ori_word_idxs = label_entity['word_idx']

            if not isinstance(ori_word_idxs[0], list):
                ori_word_idxs = [ori_word_idxs]

            for ori_word_idx in ori_word_idxs:
                token_idxs = []
                for word_idx in ori_word_idx:
                    if word_idx not in word2token_idx:
                        continue
                    token_idx = word2token_idx[word_idx]
                    if token_idx >= self.max_text_length:
                        continue
                    if len(token_idxs) == 0:
                        token_idxs.append(token_idx)
                    elif token_idx != token_idxs[-1]:
                        token_idxs.append(token_idx)

                # 这里假设label_id区分不同实体
                _entity_text.add('{}:{}'.format(label_id, '-'.join([str(v) for v in token_idxs])))
                if len(token_idxs) == 1:
                    start_idx, end_idx = token_idxs[0], token_idxs[0]
                    # _grid_labels[label_id, start_idx, end_idx] = 1
                    _grid_indexs[0].append(label_id)
                    _grid_indexs[1].append(start_idx)
                    _grid_indexs[2].append(end_idx)

                    _grid_values.append(1)


                else:
                    for i in range(len(token_idxs) - 1):
                        start_idx, end_idx = token_idxs[i], token_idxs[i + 1]
                        # _grid_labels[label_id, start_idx, end_idx] = 1
                        _grid_indexs[0].append(label_id)
                        _grid_indexs[1].append(start_idx)
                        _grid_indexs[2].append(end_idx)

                        _grid_values.append(1)

        # result['grid_labels'] = torch.LongTensor(_grid_labels)

        # sparse 无法直接输入给dataloader ：https://github.com/pytorch/pytorch/issues/20248
        # 直接用一个json来存储
        result['grid_labels'] = json.dumps({
            'indices': _grid_indexs,
            'values': _grid_values,
            'size': (
                self.ner_labels.num_classes,
                self.max_text_length,
                self.max_text_length)
        })
        # result['grid_labels'] = torch.sparse_coo_tensor(indices=_grid_indexs,
        #                                                 values=_grid_values,
        #                                                 size=(
        #                                                     self.ner_labels.num_classes,
        #                                                     self.max_text_length,
        #                                                     self.max_text_length)
        #                                                 )
        result['entity_labels'] = ' '.join(list(_entity_text))

        return result

    def process(self, sample):
        result = dict()

        # 加载json文本
        self.process_json(sample)
        # 数据增强
        self.process_augment(sample)
        # 处理文本模态特征
        text_fea_result, token2word_info, word2token_idx = self.process_text(sample)
        text_fea_result['token2word_info'] = json.dumps(token2word_info, ensure_ascii=False)
        # 处理label
        label_fea_result = self.process_label(sample, word2token_idx)

        result.update(text_fea_result)
        result.update(label_fea_result)

        return result
