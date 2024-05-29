
import json

from datasets import ClassLabel
from transformers import AutoTokenizer

from src.dataset.feature_processor.base_processor import CodeDocBaseProcessor
from src.modules.doc_utils import *


class CodeDocForTPPGP2NerProcessor(CodeDocBaseProcessor):
    '''
    CodeDocForTPPGP2NerProcessor
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
        '''
        result = dict()
        json_obj = sample['json']

        # TPP2NER label #
        _grid_labels = np.zeros((self.ner_labels.num_classes, self.max_text_length, self.max_text_length), dtype=np.int)
        _entity_text = set()

        label_entities = json_obj.get('label_entities',[])
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
                    _grid_labels[label_id, start_idx, end_idx] = 1
                else:
                    for i in range(len(token_idxs) - 1):
                        start_idx, end_idx = token_idxs[i], token_idxs[i + 1]
                        _grid_labels[label_id, start_idx, end_idx] = 1

        result['grid_labels'] = torch.LongTensor(_grid_labels)
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


        # print(result.keys())
        # for k in result.keys():
        #     print(k)
        #     try:
        #         print(result[k].shape)
        #     except Exception as e:
        #         print(type(result[k]))
        # print(f"ori_length: {result['ori_length']}")
        # for i in range(result['grid_labels'].shape[0]):
        #     print(f'Entity type: {i}')
        #     for j in range(result['grid_labels'].shape[1]):
        #         for k in range(result['grid_labels'].shape[2]):
        #             if result['grid_labels'][i][j][k] == 1:
        #                 print(j, k)
        #     for j in range(result['grid_labels'].shape[1]):
        #         for k in range(result['grid_labels'].shape[2]):
        #             if result['grid_labels'][i][j][k] != 0 and result['grid_labels'][i][j][k] != 1:
        #                 print(j, k, result['grid_labels'][i][j][k])
        #                 input()
        # print(result['grid_labels'][3])
        # print(result['grid_labels'][3].sum())
        # print(result['attention_mask'])
        # print(result['attention_mask'].sum())
        # input()

        return result
