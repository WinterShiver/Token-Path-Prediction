
import json
import math
import random

from numpy import clip
from transformers import AutoTokenizer

from src.dataset.feature_processor.base_processor import CodeDocBaseProcessor
from src.modules.doc_utils import *
from src.modules.utils import distance


class CodeDocForTPPReadingOrderTLProcessor(CodeDocBaseProcessor):
    '''
    CodeDocForTPPReadingOrderTLProcessor
    文本+Layout
    '''

    def __init__(self,

                 tokenizer,

                 max_text_length=512,

                 norm_bbox_width=1000,
                 norm_bbox_height=1000,

                 use_global_1d=False,
                 rate_shuffle_segments=0.,
                 shuffle_when_evaluation=False,
                 augment_processors=None,

                 is_train=False,
                 mask_prob=0.0,
                 **kwargs,
                 ):

        super().__init__(tokenizer=tokenizer,
                         max_text_length=max_text_length,
                         box_level='segment',
                         norm_bbox_height=norm_bbox_height,
                         norm_bbox_width=norm_bbox_width,
                         use_global_1d=use_global_1d,
                         augment_processors=augment_processors,
                         **kwargs)

        self.is_train = is_train
        self.rate_shuffle_segments = rate_shuffle_segments
        self.shuffle_when_evaluation = shuffle_when_evaluation
        self.mask_prob = mask_prob
        self.dummy_bbox = [0, 0, 0, 0, 0, 0]

    def process_label(self, sample, seg_se_map):
        '''
        生成下游任务Label
        '''
        result = dict()
        json_obj = sample['json']

        has_order_label = 'label_segment_order' in json_obj and len(json_obj['label_segment_order']) > 0

        assert (self.is_train and has_order_label) or not self.is_train, json.dumps(sample)

        if not has_order_label:
            return result

        # TPP2NER label #
        _grid_labels = np.zeros((1, self.max_text_length, self.max_text_length), dtype=np.int)

        id_set = seg_se_map.keys()

        label_segment_order = json_obj['label_segment_order']

        assert isinstance(label_segment_order[0], int), 'label_segment_order\'s type must be int'
        assert isinstance(json_obj['document'][0]['id'], int), 'sgement id\'s type must be int'

        label_segment_order = [0] + [i + 1 for i in label_segment_order if i + 1 in id_set] + [0]

        # 生成 token 2 token path
        path_pair = []
        for i in range(len(label_segment_order) - 1):
            seg_start_id = label_segment_order[i]
            seg_end_id = label_segment_order[i + 1]

            start_token_idx = seg_se_map[seg_start_id]['end_idx']
            end_token_idx = seg_se_map[seg_end_id]['start_idx']

            path_pair.append('{}_{}'.format(seg_start_id, seg_end_id))
            _grid_labels[0][start_token_idx][end_token_idx] = 1

        result['grid_labels'] = torch.LongTensor(_grid_labels)
        result['path_pair'] = '-'.join(path_pair)

        return result

    def points_process(self, box, width, height, norm_width, norm_height):
        '''
        将4点坐标转为2点坐标
        这里直接使用左上、右下2点

        '''
        # 4点坐标不是完全的矩形而是和矩形很接近的四边形，所以使用左上0，右上1，右下2来近似计算框的h和w
        # h和w作为输入，用于模型CodeLayout，通过_calc_spatial_position_embeddings计算box的h, w对应的相对位置编码
        
        # 4点坐标
        x0, y0 = box[0]
        x1, y1 = box[1]
        x2, y2 = box[2]

        x0 = clip(0, int((x0 / width) * norm_width), norm_width)
        x1 = clip(0, int((x1 / width) * norm_width), norm_width)
        x2 = clip(0, int((x2 / width) * norm_width), norm_width)

        y0 = clip(0, int((y0 / height) * norm_height), norm_height)
        y1 = clip(0, int((y1 / height) * norm_height), norm_height)
        y2 = clip(0, int((y2 / height) * norm_height), norm_height)

        w = int(round(distance([x0, y0], [x1, y1])))
        h = int(round(distance([x1, y1], [x2, y2])))

        w = min(w, norm_width)
        h = min(h, norm_height)

        box = [x0, y0, x2, y2, h, w]

        return box

    def process_text(self, sample, return_ref=False, return_tensor=True):
        '''
        处理文本模态特征
        '''
        json_obj = sample['json']

        width = json_obj['img']['width']
        height = json_obj['img']['height']
        document = json_obj['document']

        segment_idxs = []

        ## CLS Token ##
        (input_ids,
         attention_mask,
         bboxes,
         global_1d_positions,
         local_1d_positions) = [self.tokenizer.cls_token_id], [1], [self.dummy_bbox], [
            self.dummy_global_1d_position], [self.dummy_local_1d_position]

        segment_idxs.append(0)

        ref = []

        for segment_id, segment in enumerate(document):

            '''
            输入支持4点，统一转为2点坐标
            '''
            segment_box = self.points_process(segment['box'],
                                              width,
                                              height,
                                              self.norm_bbox_width,
                                              self.norm_bbox_height)
            segment_idx = segment['id']

            segment_texts = norm_text(segment['text'])

            # 分词
            tokenized_results = self.tokenizer(
                segment_texts,
                truncation=False,
                add_special_tokens=False,
                return_offsets_mapping=True,
                return_attention_mask=False,
                is_split_into_words=False
            )

            # 获取 token 和 word的映射 ;
            seg_tokens = tokenized_results.tokens()

            '''
            这里修正token，去除所有单独的▁
            存在中文情况会出现单独的▁
            '''
            space_id = self.tokenizer.convert_tokens_to_ids(['▁'])[0]
            seg_tokens = [t for t in seg_tokens if t != '▁']

            seg_input_ids = [b for b in tokenized_results['input_ids'] if b != space_id]
            seg_offset_mappings = [a for a, b in
                                   zip(tokenized_results['offset_mapping'], tokenized_results['input_ids']) if
                                   b != space_id]

            seg_local_1d_idx = 0  # seg token idx

            for seg_token, seg_input_id, seg_offset_mapping in zip(seg_tokens, seg_input_ids, seg_offset_mappings):

                if len(input_ids) >= self.max_text_length:
                    break

                input_ids.append(seg_input_id)
                attention_mask.append(1)
                global_1d_positions.append(len(global_1d_positions))
                local_1d_positions.append(seg_local_1d_idx)
                segment_idxs.append(segment_idx + 1)

                # 处理box
                bboxes.append(segment_box)
                seg_local_1d_idx += 1

        assert len(input_ids) == len(bboxes)
        assert len(input_ids) == len(local_1d_positions)
        assert len(input_ids) == len(attention_mask)

        ## Truncation
        input_ids = input_ids[:self.max_text_length]
        attention_mask = attention_mask[:self.max_text_length]
        bboxes = bboxes[:self.max_text_length]
        local_1d_positions = local_1d_positions[:self.max_text_length]
        # token2word_info = token2word_info[:self.max_text_length]

        segment_idxs = segment_idxs[:self.max_text_length]

        ori_length = len(input_ids)

        # mask 随机mask掉一部分token
        if self.is_train and self.mask_prob > 0.0:
            mask_idxs = list(range(1, ori_length))
            mask_idxs = random.choices(mask_idxs, k=int(len(mask_idxs) * self.mask_prob))

            for mask_idx in mask_idxs:
                input_ids[mask_idx] = self.tokenizer.mask_token_id

        ## Padding
        pad_len = self.max_text_length - len(input_ids)
        input_ids += [self.tokenizer.pad_token_id] * pad_len
        attention_mask += [0] * pad_len
        bboxes += [self.dummy_bbox] * pad_len
        local_1d_positions += [self.dummy_local_1d_position] * pad_len
        global_1d_positions += [self.dummy_global_1d_position] * pad_len

        result = dict()

        if return_tensor:
            result['input_ids'] = torch.tensor(input_ids)
            result['attention_mask'] = torch.tensor(attention_mask)
            result['position_1d'] = torch.tensor(global_1d_positions) if self.use_global_1d else torch.tensor(
                local_1d_positions)
            result['position_2d'] = torch.tensor(bboxes)
        else:
            result['input_ids'] = input_ids
            result['attention_mask'] = attention_mask
            result['position_1d'] = global_1d_positions if self.use_global_1d else local_1d_positions
            result['position_2d'] = bboxes

        if return_ref:
            result['chinese_ref'] = ref
        # 其他非训练数据
        result['uid'] = json_obj['uid']
        result['ori_length'] = ori_length  # 实际内容长度(包含开头[CLS])

        # 每一个segment有一个Start/End token，对应第一个和最后一个token

        seg_se_map = dict()
        for token_idx, segment_idx in enumerate(segment_idxs):
            if segment_idx not in seg_se_map:
                seg_se_map[segment_idx] = {
                    'start_idx': 9999,
                    'end_idx': -1
                }
            start_idx = seg_se_map[segment_idx]['start_idx']
            end_idx = seg_se_map[segment_idx]['end_idx']

            if token_idx > end_idx:
                seg_se_map[segment_idx]['end_idx'] = token_idx
            if token_idx < start_idx:
                seg_se_map[segment_idx]['start_idx'] = token_idx
        result['seg_se_map'] = seg_se_map

        # Added: segment global id ~ word global ids的全部映射关系，测试时使用此信息把segments排序结果转为words排序结果，进行word-level BLEU和ARD的计算

        segment_word_ids = dict()
        for segment in document:
            segment_word_ids[segment['id']] = [
                word['id'] for word in segment['words'] if 'id' in word]
        result['segment_word_ids'] = json.dumps(segment_word_ids)

        return result

    def filter(self, sample):
        '''
        过滤删除label中没有的box
        '''
        json_obj = sample['json']

        has_order_label = 'label_segment_order' in json_obj and len(json_obj['label_segment_order']) > 0

        if not has_order_label:
            return

        label_segment_order = json_obj['label_segment_order']
        segments = json_obj['document']

        segments_new = []
        for segment in segments:
            # Added: 我觉得形如{"text": " "}这样空的word很奇怪，但是这些空的word是用来表示word间隔的，去掉这部分word之后会报错，所以不动了
            # segment['words'] = [word for word in segment['words'] if not ('box' in word and 'id' in word and 'text' in word)]
            seg_id = segment['id']
            if seg_id in label_segment_order:
                segments_new.append(segment)        
        json_obj['document'] = segments_new

    def shuffle_segments(self, sample):
        random.shuffle(sample['json']['document'])

    def convert_points(self, sample):
        '''
        统一转为4点坐标
        '''
        obj = sample['json']

        for segment in obj['document']:
            box = segment['box']
            if isinstance(box[0], list):
                continue
            x0, y0, x2, y2 = box

            segment['box'] = [
                [x0, y0],
                [x2, y0],
                [x2, y2],
                [x0, y2]
            ]

    def process(self, sample):
        result = dict()

        # 加载json文本
        if isinstance(sample['json'], str):
            sample['json'] = json.loads(sample['json'])
        self.convert_points(sample)
        # filter
        self.filter(sample)
        # 数据增强
        self.process_augment(sample)
        # 破坏segments列表中的先后顺序，防止模型简单地通过position_1d学习segments列表中的先后顺序
        if self.is_train:
            # 按照rate_shuffle_segments确定是否shuffle
            if random.random() < self.rate_shuffle_segments:
                self.shuffle_segments(sample)
        else:
            # 按照shuffle_when_evaluation确定是否shuffle
            if self.shuffle_when_evaluation:
                self.shuffle_segments(sample)
        
        # 处理文本模态特征
        text_fea_result = self.process_text(sample)

        # 处理label
        label_fea_result = self.process_label(sample, text_fea_result['seg_se_map'])

        
        text_fea_result['seg_se_map'] = json.dumps(text_fea_result['seg_se_map'])

        result.update(text_fea_result)
        result.update(label_fea_result)

        return result


if __name__ == '__main__':
    sample = {
        'json': ''
    }

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path='')
    p = CodeDocForTPPReadingOrderTLProcessor(tokenizer=tokenizer)

    p.process(sample)
