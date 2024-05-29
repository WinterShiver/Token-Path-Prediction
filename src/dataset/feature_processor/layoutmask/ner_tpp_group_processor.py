
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

        all_entity_token_ids = set()

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
                all_entity_token_ids = all_entity_token_ids.union(token_idxs)
                # 这里假设label_id区分不同实体
                _entity_text.add('{}:{}'.format(label_id, '-'.join([str(v) for v in token_idxs])))
                if len(token_idxs) == 1:
                    start_idx, end_idx = token_idxs[0], token_idxs[0]
                    _grid_labels[label_id, start_idx, end_idx] = 1
                else:
                    for i in range(len(token_idxs) - 1):
                        start_idx, end_idx = token_idxs[i], token_idxs[i + 1]
                        _grid_labels[label_id, start_idx, end_idx] = 1

        if "label_groups" in json_obj:
            label_groups = json_obj.get("label_groups", [])
            _group_labels = np.zeros((self.max_text_length, self.max_text_length), dtype=np.int)

            _group_label_text = []
            for label_group in label_groups:
                label_group_1, label_group_2 = label_group
                # 1
                group_token_set_1 = set()
                for word_idx in label_group_1:
                    if word_idx not in word2token_idx:
                        continue
                    token_idx = word2token_idx[word_idx]
                    if token_idx >= self.max_text_length:
                        continue
                    if token_idx not in all_entity_token_ids:
                        # 分组内过滤非实体token
                        continue
                    group_token_set_1.add(token_idx)
                # 2
                group_token_set_2 = set()
                for word_idx in label_group_2:
                    if word_idx not in word2token_idx:
                        continue
                    token_idx = word2token_idx[word_idx]
                    if token_idx >= self.max_text_length:
                        continue
                    if token_idx not in all_entity_token_ids:
                        # 分组内过滤非实体token
                        continue
                    group_token_set_2.add(token_idx)
                if len(group_token_set_1) > 0 and len(group_token_set_2) > 0:
                    ordered_list_1 = sorted(list(group_token_set_1))
                    ordered_list_2 = sorted(list(group_token_set_2))
                    ordered_list = sorted(ordered_list_1 + ordered_list_2)
                    _group_label_text.append('-'.join([str(i) for i in ordered_list]))
                    for i in ordered_list_1:
                        for j in ordered_list_2:
                            _group_labels[i, j] = 1

            result['group_labels'] = torch.LongTensor(_group_labels)
            result['group_label_texts'] = ' '.join(_group_label_text)
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

        return result


class CodeDocForTPPGP2NerProcessorV2(CodeDocBaseProcessor):
    '''
    CodeDocForTPPGP2NerProcessorV2
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

        将所有有关联的实体全连接

        '''
        result = dict()
        json_obj = sample['json']

        if "label_groups" in json_obj:

            entity_type_map = dict()

            _group_labels = np.zeros((self.max_text_length, self.max_text_length), dtype=np.int)

            label_entities = json_obj.get('label_entities', [])
            entity_set = []
            for label_entity in label_entities:
                word_idxs = label_entity['word_idx']

                for word_idx in word_idxs:
                    token_idx = set([])
                    for w_id in word_idx:
                        if w_id not in word2token_idx:
                            continue
                        t_id = word2token_idx[w_id]
                        if t_id >= self.max_text_length:
                            continue
                        token_idx.add(t_id)

                    if len(token_idx) == 0:
                        continue

                    token_idx = sorted(list(token_idx))

                    entity_set.append(token_idx)

                    k = '-'.join([str(v) for v in token_idx])
                    entity_type_map[k] = label_entity['label']

            label_groups = json_obj.get("label_groups", [])
            _group_label_text = []
            for label_group in label_groups:
                label_group_1, label_group_2 = label_group
                # 1
                group_token_set_1 = set()
                for word_idx in label_group_1:
                    if word_idx not in word2token_idx:
                        continue
                    token_idx = word2token_idx[word_idx]
                    if token_idx >= self.max_text_length:
                        continue
                    group_token_set_1.add(token_idx)
                # 2
                group_token_set_2 = set()
                for word_idx in label_group_2:
                    if word_idx not in word2token_idx:
                        continue
                    token_idx = word2token_idx[word_idx]
                    if token_idx >= self.max_text_length:
                        continue
                    group_token_set_2.add(token_idx)

                if len(group_token_set_1) > 0 and len(group_token_set_2) > 0:
                    group_token_set_1 = sorted(list(group_token_set_1))
                    group_token_set_2 = sorted(list(group_token_set_2))

                    ordered_list_1 = '-'.join([str(v) for v in group_token_set_1])
                    ordered_list_2 = '-'.join([str(v) for v in group_token_set_2])

                    _group_label_text.append(ordered_list_1 + ':' + ordered_list_2)

                    for i in group_token_set_1:
                        for j in group_token_set_2:
                            _group_labels[i, j] = 1
                            _group_labels[j, i] = 1

            result['group_labels'] = torch.LongTensor(_group_labels)
            result['group_label_texts'] = ' '.join(_group_label_text)

            # result['grid_labels'] = torch.LongTensor(_grid_labels)
            result['entities'] = json.dumps(entity_set)
            result['entity_type_map'] = json.dumps(entity_type_map)

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
