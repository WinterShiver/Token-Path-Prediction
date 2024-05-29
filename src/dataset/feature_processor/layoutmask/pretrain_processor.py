import json
import random
import numpy as np
import torch
from torchvision import transforms
from transformers import AutoTokenizer

from src.dataset.feature_processor.base_processor import CodeDocBaseProcessor
from src.modules.doc_utils import is_sentence_piece_subword_v3
from src.modules.utils import points4to2, box_norm


class LayoutmaskPretrainProcessor(CodeDocBaseProcessor):
    '''
    LayoutMask 预训练任务数据处理
    '''
    def __init__(self,
                 tokenizer,
                 max_text_length=512,
                 box_level="segment",  # or word
                 norm_bbox_width=1000,
                 norm_bbox_height=1000,
                 use_global_1d=False,
                 augment_processors=None,
                 **kwargs,
                 ):

        #super().__init__(self)
        self.tokenizer = tokenizer

        self.norm_bbox_width = norm_bbox_width
        self.norm_bbox_height = norm_bbox_height
        self.box_level = box_level
        self.max_text_length = max_text_length

        self.use_global_1d = use_global_1d

        self.augment_processors = augment_processors

        self.dummy_bbox = [0, 0, 0, 0]
        self.dummy_local_1d_position = 0
        self.dummy_global_1d_position = 0

        self.mpm_prob = kwargs["mpm_prob"]

    def process_text(self, sample):
        '''
        处理文本模态特征
        '''
        json_obj = sample['json']

        self.width = json_obj['img']['width']
        self.height = json_obj['img']['height']
        document = json_obj['document']

        #mpm预处理
        document = self.process_for_mpm(document)

        ## CLS Token ## 
        input_ids = [self.tokenizer.cls_token_id]
        attention_mask = [1]
        bboxes = [self.dummy_bbox]
        global_1d_positions = [self.dummy_global_1d_position]
        local_1d_positions = [self.dummy_local_1d_position]
        attention_mask_mpm = [0]
        position_gt_mpm = [self.dummy_bbox] 
        ref = []

        for segment_id, segment in enumerate(document):
            '''
            输入支持4点，统一转为2点坐标
            '''
            segment_box = points4to2(segment['box'])

            if segment['mpm_flag'] == 1: 
                norm_segment_box = segment_box #直接使用原始box，因为已经是[0,0,0,x]的情况了
            else:
                norm_segment_box = box_norm(
                    segment_box,
                    self.width,
                    self.height,
                    self.norm_bbox_width,
                    self.norm_bbox_height
                )

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

                bboxes.append(norm_segment_box)
                seg_local_1d_idx += 1
                # ref用于模拟word wwm
                if last_token and is_sentence_piece_subword_v3(seg_token, last_token):
                    ref.append(len(input_ids) - 1)

                last_token = seg_token
            
            segment_token_length = len(seg_input_ids)

            #获取MPM相关参数：
            attention_mask_mpm.extend([segment['mpm_flag']] * segment_token_length)
            mpm_box_gt = [float(norm_segment_box[0])/self.norm_bbox_width, 
                          float(norm_segment_box[1])/self.norm_bbox_height,
                          float(norm_segment_box[2])/self.norm_bbox_width, 
                          float(norm_segment_box[3])/self.norm_bbox_height]
            position_gt_mpm.extend([mpm_box_gt] * segment_token_length)

        assert len(input_ids) == len(bboxes)
        assert len(input_ids) == len(local_1d_positions)
        assert len(input_ids) == len(attention_mask)

        ## Truncation
        input_ids = input_ids[:self.max_text_length]
        attention_mask = attention_mask[:self.max_text_length]
        bboxes = bboxes[:self.max_text_length]
        local_1d_positions = local_1d_positions[:self.max_text_length]
        attention_mask_mpm = attention_mask_mpm[:self.max_text_length]
        position_gt_mpm = position_gt_mpm[:self.max_text_length]

        ## Padding
        pad_len = self.max_text_length - len(input_ids)
        input_ids += [self.tokenizer.pad_token_id] * pad_len
        attention_mask += [0] * pad_len
        bboxes += [self.dummy_bbox] * pad_len
        local_1d_positions += [self.dummy_local_1d_position] * pad_len
        global_1d_positions += [self.dummy_global_1d_position] * pad_len
        attention_mask_mpm += [0] * pad_len
        position_gt_mpm += [self.dummy_bbox] * pad_len

        result = dict()
        result['input_ids'] = torch.tensor(input_ids)
        result['attention_mask'] = torch.tensor(attention_mask)
        result['position_1d'] = torch.tensor(global_1d_positions) if self.use_global_1d else torch.tensor(
            local_1d_positions)
        result['position_2d'] = torch.tensor(bboxes)
        result['mpm_attention_mask'] = torch.tensor(attention_mask_mpm)
        result['mpm_labels'] = torch.tensor(position_gt_mpm)
        result['chinese_ref'] = ref

        return result

    def process(self, sample):
        # 加载json文本特征
        self.process_json(sample)
        # 数据增强
        self.process_augment(sample)
        # 处理文本模态特征
        text_result = self.process_text(sample)

        return text_result

    def process_for_mpm(self, segments):
        new_segments = []
        self.mpm_words = []  #已经masked的词表
        for seg in segments:
            new_segments.extend(self.mpm_for_single_segment(seg))

        num = 0
        mpm_word_ids = [i for i in range(self.norm_bbox_height)]
        random.shuffle(mpm_word_ids)
        for seg in new_segments:
            if seg['mpm_flag'] == 1:
                seg['box'] = [0,0,0,mpm_word_ids[min(num, len(mpm_word_ids)-1)]]
                num = num+1
        return new_segments

    def mpm_for_single_segment(self, segment):
        #对于每个word，按照args.mpm_prob的概率进行切分
        words_num = len(segment['words'])
        thrd = self.mpm_prob
        output = []

        probs = np.random.random(words_num)
        words_cur = []

        for i in range(len(probs)):
            if probs[i] > thrd or segment['words'][i]['text'] in self.mpm_words:  #不能有重复的words
                words_cur.append(segment['words'][i]) 
            else:
                #将不被masked的word组成新的pre-segment
                if len(words_cur)>0:
                    seg_new = self.words2segment(words_cur)
                    seg_new['mpm_flag'] = 0
                    seg_new['mpm_box_gt'] = seg_new['box']
                    output.append(seg_new)
                    words_cur = []
                # MPM选中的word的原始box要被抹去
                seg_new = self.words2segment(segment['words'][i:i+1])
                seg_new['mpm_flag'] = 1
                seg_new['mpm_box_gt'] = seg_new['box']  # 真值被替换掉;数值有范围
                seg_new['box'] = self.dummy_bbox  # 输入的box要被mask 
                output.append(seg_new)      
                self.mpm_words.append(segment['words'][i]['text'])   
        # 将不被masked的word组成新的after-segment
        if len(words_cur)>0:
            seg_new = self.words2segment(words_cur)
            seg_new['mpm_flag'] = 0
            seg_new['mpm_box_gt'] = seg_new['box']
            output.append(seg_new)        

        assert sum([len(p['words']) for p in output]) == len(segment['words'])
        assert len(self.mpm_words) == len(set(self.mpm_words)), print(self.mpm_words)
        
        return output

    def words2segment(self, words):
        text = " ".join([word['text'] for word in words])
        box = [min([word['box'][0] for word in words]),
            min([word['box'][1] for word in words]),
            max([word['box'][2] for word in words]),
            max([word['box'][3] for word in words])]
        box = [min(box[0], box[2]),
                min(box[1], box[3]),
                max(box[0], box[2]),
                max(box[1], box[3])]    
        return {'box': box,
               'text': text,
                'words': words}
    
if __name__ == '__main__':
    from src.dataset.code_doc_collator import DocumentWwmMlmCollator
    from src.dataset.code_doc_dataset import CodeDocLocalDataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained('/models/tokenizers/chinese')
    print(tokenizer.cls_token_id)

    collate_fn_train = DocumentWwmMlmCollator(
        tokenizer=tokenizer,
        mlm_prob=0.15,
        use_chs_wwm=False
    )
    
    # 数据集和预处理方法
    print('开始加载数据和预处理')
    augment_processors = []
    #if self.args.enable_aug:
    #    augment_processors.append(AugForSegmentSplit(prob=0.2))
    #    augment_processors.append(AugForRndMove(prob=0.95))
    data_processor_train = LayoutmaskPretrainProcessor(tokenizer,
                                                        max_text_length=512,
                                                        norm_bbox_height=1000,
                                                        norm_bbox_width=1000,
                                                        augment_processors=augment_processors,
                                                        mpm_prob= 0.1,
                                                        )
    train_dataset = CodeDocLocalDataset(data_dir="",
                                        data_processor=data_processor_train,
                                        dataset_name="data.test.txt",
                                        shuffle= False
                                        )
    print(len(train_dataset))
    print(train_dataset[0])