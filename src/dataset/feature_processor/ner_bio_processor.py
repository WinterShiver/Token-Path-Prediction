
import json
from src.dataset.feature_processor.base_processor import CodeDocBaseProcessor
from src.modules.doc_utils import *


class CodeDocForBIO2NerProcessor(CodeDocBaseProcessor):
    '''
    CodeDocForBIO2NerProcessor
    '''

    def __init__(self,
                 tokenizer,

                 ner_labels,
                 default_label='O',

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
        self.default_label = default_label

    def process_label(self, sample, word2token_idx, ori_length):
        '''
        生成下游任务Label
        '''
        result = dict()
        json_obj = sample['json']
        #has_entity_label = 'label_entities' in json_obj and len(json_obj['label_entities']) > 0
        has_entity_label = 'label_entities' in json_obj
        if not has_entity_label:
            #print('No label entity is found!')
            return result

        word_id2entity_id = dict()

        label_entities = json_obj['label_entities']
        '''
        这里加入各种后缀的目的是为了区分不同segment、或者同一segment内连续相同的label
        '''
        for label_entity in label_entities:
            label = label_entity['label']
            word_idxs = label_entity['word_idx']
            entity_id = label_entity['entity_id']
            label_with_id = '{}▁{}'.format(label, entity_id)
            for i, word_idx in enumerate(word_idxs):
                if isinstance(word_idx, list):
                    # 存在多个片段
                    label_with_id = '{}_{}'.format(label_with_id, i)
                    for wid in word_idx:
                        word_id2entity_id[wid] = label_with_id
                else:
                    word_id2entity_id[word_idx] = label_with_id
        # 加上segment标，区分不同segment的同类label
        # for segment_id, segment in enumerate(json_obj['document']):
        #     for word in segment['words']:
        #         word_id = word['id']
        #         if word_id in word_id2entity_id:
        #             word_label = '{}_{}'.format(word_id2entity_id[word_id], segment_id)
        #             word_id2entity_id[word_id] = word_label

        # 生成token的label list
        labels = [self.default_label for _ in range(self.max_text_length)]
        label_ids = [-100 for _ in range(self.max_text_length)]

        # 针对可能存在的离散实体，生成一个可以直接在metric里面和预测结果比的标签
        result['labels_str'] = []
        for label_entity in label_entities:
            label = label_entity['label']
            word_idxs = label_entity['word_idx']
            entity_id = label_entity['entity_id']
            label_with_id = '{}▁{}'.format(label, entity_id)
            begin, end = None, None
            begin_token_idx, end_token_idx = None, None
            spans = []
            for i, word_idx in enumerate(word_idxs):
                if begin is None:
                    begin, end = word_idx, word_idx
                    if word_idx in word2token_idx:
                        begin_token_idx, end_token_idx = word2token_idx[word_idx], word2token_idx[word_idx]
                elif word_idx == end + 1:
                    end = word_idx
                    if word_idx in word2token_idx:
                        if begin_token_idx is None:
                            begin_token_idx, end_token_idx = word2token_idx[word_idx], word2token_idx[word_idx]
                        else:
                            end_token_idx =  word2token_idx[word_idx]
                else:
                    if not (begin_token_idx is None):
                        spans.append([begin_token_idx, end_token_idx])
                    begin, end = word_idx, word_idx
                    if word_idx in word2token_idx:
                        begin_token_idx, end_token_idx = word2token_idx[word_idx], word2token_idx[word_idx]
                    else:
                        begin_token_idx, end_token_idx = None, None
            else:
                if not (begin_token_idx is None):
                    spans.append([begin_token_idx, end_token_idx])
            result['labels_str'].append(
                f'{label}-' + '-'.join([f'{begin_token_idx-1}-{end_token_idx-1}' for begin_token_idx, end_token_idx in spans]))
        result['labels_str'] = json.dumps(result['labels_str'])

        for word_id, token_id in word2token_idx.items():
            if word_id not in word_id2entity_id:
                continue
            word_label = word_id2entity_id[word_id]
            labels[token_id] = word_label
        # 转为bio
        labels_bio = [v.split('▁')[0] for v in labels2bio(labels, self.default_label)]
        # 更新label_ids
        for idx in range(1, ori_length):  # 0是CLS
            label_ids[idx] = self.ner_labels.str2int(labels_bio[idx])

        result['labels'] = torch.tensor(label_ids)

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
        label_fea_result = self.process_label(sample, word2token_idx, ori_length=text_fea_result['ori_length'])

        result.update(text_fea_result)
        result.update(label_fea_result)

        return result
