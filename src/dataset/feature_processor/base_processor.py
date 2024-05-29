import json
import time 

import torch
from torchvision import transforms
from torchvision.datasets.folder import pil_loader
from torchvision.transforms import Compose
from transformers import AutoTokenizer

from src.modules.doc_utils import *
from src.modules.image_utils import RandomResizedCropAndInterpolationWithTwoPic


class CodeDocBaseProcessor:
    '''
    CodeDocBaseProcessor
    '''

    def __init__(self,
                 tokenizer,

                 max_text_length=512,

                 box_level="segment",  # or word

                 norm_bbox_width=1000,
                 norm_bbox_height=1000,

                 use_global_1d=False,

                 augment_processors=None,

                 use_image=False,
                 image_input_size=224,

                 **kwargs,
                 ):

        self.tokenizer = tokenizer

        self.norm_bbox_width = norm_bbox_width
        self.norm_bbox_height = norm_bbox_height
        self.box_level = box_level
        self.max_text_length = max_text_length

        self.use_global_1d = use_global_1d

        self.augment_processors = augment_processors

        self.use_image = use_image
        self.image_input_size = image_input_size

        if use_image:
            IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
            IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
            IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
            IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)

            imagenet_default_mean_and_std = False
            mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
            std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD
            self.common_transform = Compose([
                # transforms.ColorJitter(0.4, 0.4, 0.4),
                # transforms.RandomHorizontalFlip(p=0.5),
                RandomResizedCropAndInterpolationWithTwoPic(
                    size=image_input_size, interpolation='bicubic'),
            ])

            self.patch_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=torch.tensor(mean),
                    std=torch.tensor(std))
            ])

            self.IMAGE_LEN = int(image_input_size / 16) * int(image_input_size / 16) + 1

        self.dummy_bbox = [0, 0, 0, 0]
        self.dummy_local_1d_position = 0
        self.dummy_global_1d_position = 0

    def process_text(self, sample, return_ref=False, return_tensor=True):
        '''
        处理文本模态特征
        '''
        json_obj = sample['json']

        width = json_obj['img']['width']
        height = json_obj['img']['height']
        document = json_obj['document']
        token2word_info = []
        word2token_idx = dict()

        ## CLS Token ##
        (input_ids,
         attention_mask,
         bboxes,
         global_1d_positions,
         local_1d_positions) = [self.tokenizer.cls_token_id], [1], [self.dummy_bbox], [
            self.dummy_global_1d_position], [self.dummy_local_1d_position]

        token2word_info.append({})

        ref = []

        for segment_id, segment in enumerate(document):

            '''
            输入支持4点，统一转为2点坐标
            '''
            segment_box = points4to2(segment['box'])

            segment_word_ids = []
            segment_word_texts = []
            segment_word_boxs = []

            for word in segment['words']:
                word_id = word['id']
                word_text = word['text']
                word_box = word['box'] if ('box' in word and len(word['box']) > 0) else segment_box
                word_box = points4to2(word_box)

                segment_word_ids.append(word_id)
                segment_word_texts.append(word_text)
                segment_word_boxs.append(word_box)

            segment_texts = ''.join([norm_text(v) for v in segment_word_texts])

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
            pre_offset_mapping = None
            last_token = None
            for seg_token, seg_input_id, seg_offset_mapping in zip(seg_tokens, seg_input_ids, seg_offset_mappings):

                if len(input_ids) >= self.max_text_length:
                    break
                '''
                针对一个字符切为多个字符的情况，其多个seg_offset_mapping会指向同一个char idx，这里只保留第一个
                注意：在文档真实场景，OCR后的文字一般不会有这种情况
                '''
                if pre_offset_mapping and (
                        seg_offset_mapping[0] == pre_offset_mapping[0]
                        and seg_offset_mapping[1] == pre_offset_mapping[1]):
                    continue
                
                # token 到 word id映射(segment内)；一对多
                token_idx = len(input_ids)
                token_word_idx = range(seg_offset_mapping[0], seg_offset_mapping[1])

                # token->word info
                try:
                    token_word_ids = [segment_word_ids[v] for v in token_word_idx]
                    token_word_texts = [segment_word_texts[v] for v in token_word_idx]
                    token_word_boxs = [segment_word_boxs[v] for v in token_word_idx]
                except Exception as e:
                    print(sample['json']['uid'])
                    print(token_word_idx)
                    assert len(segment_word_ids) == len(segment_word_texts) == len(segment_word_boxs)
                    print(len(segment_word_ids))
                    raise e

                ### 返回 word - token 映射信息 ###
                token2word_info.append({
                    'token': seg_token,
                    'word_ids': token_word_ids,
                    'word_texts': token_word_texts,
                    'word_boxs': token_word_boxs
                })
                for token_word_id in token_word_ids:
                    word2token_idx[token_word_id] = token_idx

                ### 返回 word - token 映射信息 ###
                input_ids.append(seg_input_id)
                attention_mask.append(1)
                global_1d_positions.append(len(global_1d_positions))
                local_1d_positions.append(seg_local_1d_idx)

                # print(seg_token)
                # print(seg_input_id)
                # print(token_idx)
                # print(token_word_idx)
                # print(token_word_ids)
                # print(token_word_texts)
                # print(token_word_boxs)
                # time.sleep(1.5)

                # 处理box
                if self.box_level == 'word':
                    word_box = merge_bbox(token_word_boxs)
                    if word_box is None:
                        word_box = segment_box
                    # 合并对应的word的box
                    norm_word_box = box_norm(
                        word_box,
                        width,
                        height,
                        self.norm_bbox_width,
                        self.norm_bbox_height
                    )
                    bboxes.append(norm_word_box)
                else:
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
                if return_ref and last_token and is_sentence_piece_subword_v3(seg_token, last_token):
                    ref.append(len(input_ids) - 1)

                last_token = seg_token
                pre_offset_mapping = seg_offset_mapping

        assert len(input_ids) == len(bboxes)
        assert len(input_ids) == len(local_1d_positions)
        assert len(input_ids) == len(attention_mask)

        ## Truncation
        input_ids = input_ids[:self.max_text_length]
        attention_mask = attention_mask[:self.max_text_length]
        bboxes = bboxes[:self.max_text_length]
        local_1d_positions = local_1d_positions[:self.max_text_length]
        token2word_info = token2word_info[:self.max_text_length]

        ori_length = len(input_ids)

        ## Padding
        pad_len = self.max_text_length - len(input_ids)
        input_ids += [self.tokenizer.pad_token_id] * pad_len
        attention_mask += [0] * pad_len
        bboxes += [self.dummy_bbox] * pad_len
        local_1d_positions += [self.dummy_local_1d_position] * pad_len
        global_1d_positions += [self.dummy_global_1d_position] * pad_len
        token2word_info += [{}] * pad_len

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

        return result, token2word_info, word2token_idx

    def process_image(self, sample):
        result = {}

        ipath = sample["image"]
        img = pil_loader(ipath)
        for_patches, _ = self.common_transform(img)
        patch = self.patch_transform(for_patches)
        visual_attention_mask = torch.ones((self.IMAGE_LEN,), dtype=torch.long)

        result['images'] = patch
        result['visual_attention_mask'] = visual_attention_mask
        return result

    def process_json(self, sample):

        json_path_or_obj = sample['json']
        '''
        json_path_or_obj: json路径，json string 或者 json obj
        '''
        if isinstance(json_path_or_obj, str):
            if json_path_or_obj.startswith('{'):
                json_obj = json.loads(json_path_or_obj)
            else:
                with open(json_path_or_obj, 'r', encoding='utf-8') as fp:
                    json_obj = json.load(fp)
        else:
            json_obj = json_path_or_obj

        # 兼容dict的格式
        if 'label_entities' in json_obj:
            # 兼容 dict的格式
            label_entities = json_obj['label_entities']
            if isinstance(label_entities, dict):
                label_entities_new = []
                for k, v in label_entities.items():
                    # if k.startswith('自定义'):
                    #     continue
                    label_entity = {
                        "entity_id": v['entity_id'],
                        "label": v['label'],
                        "word_idx": v['word_idx']
                    }
                    label_entities_new.append(label_entity)
                json_obj['label_entities'] = label_entities_new

        sample['json'] = json_obj

    def process_augment(self, sample):
        '''
        数据增强
        '''
        if self.augment_processors:
            for augment_processor in self.augment_processors:
                augment_processor.process(sample)

    def process(self, sample):
        result = dict()

        # 加载json文本特征
        self.process_json(sample)
        # 数据增强
        self.process_augment(sample)
        # 处理文本模态特征
        text_fea_result, token2word_info, word2token_idx = self.process_text(sample)
        # 处理图像模态特征
        if self.use_image:
            image_result = self.process_image(sample)

            text_fea_result['attention_mask'] = torch.cat([text_fea_result['attention_mask'],
                                                           image_result['visual_attention_mask']])
            image_result.pop('visual_attention_mask')
            result.update(image_result)

        text_fea_result['token2word_info'] = json.dumps(token2word_info, ensure_ascii=False)

        result.update(text_fea_result)

        return result
