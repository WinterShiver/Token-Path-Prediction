import json

import numpy as np
import torch
from torchvision import transforms
from transformers import AutoTokenizer

from src.dataset.feature_processor.base_processor import CodeDocBaseProcessor
from src.model.code_layout.configuration_code_layout import CodeLayoutConfig
from src.modules.doc_utils import is_sentence_piece_subword_v3
from src.modules.utils import points4to2, box_norm


class CodeDocForMLMProcessor(CodeDocBaseProcessor):
    '''
    MLM预训练任务数据处理
    '''

    def process_text(self, sample):
        '''
        处理文本模态特征
        '''
        json_obj = sample['json']

        width = json_obj['img']['width']
        height = json_obj['img']['height']
        document = json_obj['document']

        ## CLS Token ##
        (input_ids,
         attention_mask,
         bboxes,
         global_1d_positions,
         local_1d_positions) = [self.tokenizer.cls_token_id], [1], [self.dummy_bbox], [
            self.dummy_global_1d_position], [self.dummy_local_1d_position]

        ref = []

        for segment_id, segment in enumerate(document):
            '''
            输入支持4点，统一转为2点坐标
            '''
            segment_box = points4to2(segment['box'])
            segment_texts = segment['text']

            # 分词
            tokenized_results = self.tokenizer(
                segment_texts,
                truncation=False,
                add_special_tokens=False,
                return_offsets_mapping=False,
                return_attention_mask=False,
                is_split_into_words=False
            )

            # 获取 token 和 word的映射 ;
            seg_tokens = tokenized_results.tokens()
            '''
            存在中文情况会出现单独的▁, 这里修正token，去除所有单独的▁
            '''
            space_id = self.tokenizer.convert_tokens_to_ids(['▁'])[0]
            seg_tokens = [t for t in seg_tokens if t != '▁']
            seg_input_ids = [b for b in tokenized_results['input_ids'] if b != space_id]

            seg_local_1d_idx = 0  # seg token idx
            last_token = None
            for seg_token, seg_input_id in zip(seg_tokens, seg_input_ids):

                if len(input_ids) >= self.max_text_length:
                    break

                ### 返回 word - token 映射信息 ###
                input_ids.append(seg_input_id)
                attention_mask.append(1)
                global_1d_positions.append(len(global_1d_positions))
                local_1d_positions.append(seg_local_1d_idx)

                norm_segment_box = box_norm(
                    segment_box,
                    width,
                    height,
                    self.norm_bbox_width,
                    self.norm_bbox_height
                )

                bboxes.append(norm_segment_box)
                seg_local_1d_idx += 1
                # ref用于模拟word wwm
                if last_token and is_sentence_piece_subword_v3(seg_token, last_token):
                    ref.append(len(input_ids) - 1)

                last_token = seg_token

        assert len(input_ids) == len(bboxes)
        assert len(input_ids) == len(local_1d_positions)
        assert len(input_ids) == len(attention_mask)

        ## Truncation
        input_ids = input_ids[:self.max_text_length]
        attention_mask = attention_mask[:self.max_text_length]
        bboxes = bboxes[:self.max_text_length]
        local_1d_positions = local_1d_positions[:self.max_text_length]

        ## Padding
        pad_len = self.max_text_length - len(input_ids)
        input_ids += [self.tokenizer.pad_token_id] * pad_len
        attention_mask += [0] * pad_len
        bboxes += [self.dummy_bbox] * pad_len
        local_1d_positions += [self.dummy_local_1d_position] * pad_len
        global_1d_positions += [self.dummy_global_1d_position] * pad_len

        result = dict()
        result['input_ids'] = torch.tensor(input_ids)
        result['attention_mask'] = torch.tensor(attention_mask)
        result['position_1d'] = torch.tensor(global_1d_positions) if self.use_global_1d else torch.tensor(
            local_1d_positions)
        result['position_2d'] = torch.tensor(bboxes)
        result['chinese_ref'] = ref

        return result

    def process(self, sample):

        # 加载json文本特征
        self.process_json(sample)

        # 处理文本模态特征
        text_result = self.process_text(sample)

        return text_result
