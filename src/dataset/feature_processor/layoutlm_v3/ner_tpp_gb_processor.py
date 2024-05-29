
import json

from datasets import ClassLabel
from transformers import AutoTokenizer

from src.dataset.feature_processor.base_processor import CodeDocBaseProcessor
from src.modules.doc_utils import *


class LayoutLMv3ForTPPGPNerProcessor(CodeDocBaseProcessor):
    '''
    LayoutLMv3ForTPPGPNerProcessor
    '''

    def __init__(self,
                 tokenizer,

                 ner_labels,

                 max_text_length=512,

                 box_level="segment",  # or word

                 norm_bbox_width=1000,
                 norm_bbox_height=1000,

                 use_global_1d=True,
                 augment_processors=None,

                 use_image=True,
                 image_input_size=224,

                 **kwargs,
                 ):

        super().__init__(tokenizer=tokenizer,
                         max_text_length=max_text_length,
                         box_level=box_level,
                         norm_bbox_height=norm_bbox_height,
                         norm_bbox_width=norm_bbox_width,
                         use_global_1d=use_global_1d,
                         augment_processors=augment_processors,
                         use_image=use_image,
                         image_input_size=image_input_size,
                         **kwargs)
        self.ner_labels = ner_labels
    
    def process_label(self, sample, word2token_idx):
        '''
        生成下游任务Label
        '''
        result = dict()
        json_obj = sample['json']

        # TPP2NER label #
        # Added: _grid_labels的大小为(num_classes, max_seq_length, max_seq_length), 其中max_seq_length是text tokens & visual tokens的总长度
        if self.max_seq_length is None:
            self.max_seq_length = self.max_text_length
        _grid_labels = np.zeros((self.ner_labels.num_classes, self.max_seq_length, self.max_seq_length), dtype=np.int)
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

        # 加载json文本特征
        self.process_json(sample)
        # 数据增强
        self.process_augment(sample)
        # 处理文本模态特征
        text_fea_result, token2word_info, word2token_idx = self.process_text(sample)
        # text_fea_result.pop('position_1d')  # LayoutLMv3的实现，自动生成1d的位置Idx，这里删掉

        # 处理图像模态特征
        if self.use_image:
            image_fea_result = self.process_image(sample)
            # Added: content mask逻辑：文字部分和attention_mask相同，视觉部分置0，下游TPP模型不检查视觉部分的输出标签
            text_fea_result['content_mask'] = torch.cat([
                text_fea_result['attention_mask'], 
                torch.zeros_like(image_fea_result['visual_attention_mask'])])
            # 更新图Patchs对应attention mask
            text_fea_result['attention_mask'] = torch.cat(
                [text_fea_result['attention_mask'], image_fea_result['visual_attention_mask']])
            image_fea_result.pop('visual_attention_mask')
            result.update(image_fea_result)
            # Added: 确定序列总长度，供process_label使用，用来确定grid_labels的维度
            self.max_seq_length = int(text_fea_result['attention_mask'].shape[0])

        text_fea_result['token2word_info'] = json.dumps(token2word_info, ensure_ascii=False)

        # 处理label
        label_fea_result = self.process_label(sample, word2token_idx) # ori_length=text_fea_result['ori_length']

        result.update(text_fea_result)
        result.update(label_fea_result)

        return result


if __name__ == '__main__':
    sample = {
        'json': '',
        'image': ''
    }

    pretrained_model_path = '/path/to/layoutlmv3'
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=pretrained_model_path)
    ner_labels = ClassLabel(
        names_file='/path/to/dataset/labels.txt')
    p = LayoutLMv3ForTPPGPNerProcessor(tokenizer=tokenizer, use_image=True, ner_labels=ner_labels)
    data = p.process(sample)
    print(data.keys())
    for k in data.keys():
        print(k)
        try:
            print(data[k].shape)
        except Exception as e:
            print(type(data[k]))
    
