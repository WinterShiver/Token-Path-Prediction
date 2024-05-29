
import torch
from typing import List, Union, Any, Dict
import jieba

from transformers import DataCollatorForWholeWordMask, BatchEncoding
import logging
from transformers.data.data_collator import _torch_collate_batch, tolist
from src.modules.utils import *

jieba.setLogLevel(logging.INFO)
logger = logging.getLogger(__name__)


class MyDataCollatorForWholeWordMask(DataCollatorForWholeWordMask):
    '''
    在原来英文wwm基础上，对连续的单字中文 应用wwm
    '''

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]], use_chs_wwm=False) -> Dict[
        str, Any]:
        if isinstance(examples[0], (dict, BatchEncoding)):
            input_ids = [e["input_ids"] for e in examples]
        else:
            input_ids = examples
            examples = [{"input_ids": e} for e in examples]

        batch_input = _torch_collate_batch(input_ids, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)

        mask_labels = []
        for e in examples:
            ref_tokens = []
            for id in tolist(e["input_ids"]):
                token = self.tokenizer._convert_id_to_token(id)
                ref_tokens.append(token)

            # For Chinese tokens, we need extra inf to mark sub-word, e.g [喜,欢]-> [喜，##欢]
            if "chinese_ref" in e:
                ref_pos = tolist(e["chinese_ref"])
                len_seq = len(e["input_ids"])
                for i in range(len_seq):
                    if i in ref_pos:
                        ref_tokens[i] = "##" + ref_tokens[i]
            '''
            如果是连续的单字中文，合并用jieba分词，增加## 用于wwm
            '''
            if use_chs_wwm:
                chs_tokens = []
                for idx in range(len(ref_tokens) + 1):
                    ref_token = ref_tokens[idx] if idx < len(ref_tokens) else None
                    if ref_token and len(ref_token) == 1 and is_chinese(ref_token):
                        chs_tokens.append([ref_token, idx])
                    else:
                        if len(chs_tokens) > 1:
                            text = ''.join([chs_token[0] for chs_token in chs_tokens])
                            word_i = 0
                            for word in jieba.cut(text):
                                if len(word) > 1:
                                    for word_j in range(1, len(word)):
                                        chs_token_i = word_i + word_j
                                        ref_token_i = chs_tokens[chs_token_i][1]
                                        ref_tokens[ref_token_i] = '##' + ref_tokens[ref_token_i]
                                word_i += len(word)

                        if len(chs_tokens) > 0:
                            chs_tokens = []

            mask_labels.append(self._whole_word_mask(ref_tokens))
        batch_mask = _torch_collate_batch(mask_labels, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
        inputs, labels = self.torch_mask_tokens(batch_input, batch_mask)
        return {"input_ids": inputs, "labels": labels}


class DocumentWwmMlmCollator:

    def __init__(self, tokenizer, mlm_prob=0.15, use_chs_wwm=True):
        self.tokenizer = tokenizer
        self.use_chs_wwm = use_chs_wwm
        self.data_collator = MyDataCollatorForWholeWordMask(tokenizer=tokenizer, mlm=True, mlm_probability=mlm_prob)

    def __call__(self, examples, **kwargs):
        collated = self.data_collator.torch_call(examples, use_chs_wwm=self.use_chs_wwm)

        keys = list(examples[0].keys())
        for key in keys:
            if key not in collated:
                if torch.is_tensor(examples[0][key]):
                    collated[key] = torch.stack([e[key] for e in examples], dim=0)
                else:
                    collated[key] = [e[key] for e in examples]
        return collated


class DocumentCollator:

    def __init__(self):
        pass

    def __call__(self, examples, **kwargs):
        
        keys = list(examples[0].keys())
        for key in keys:
            if key not in collated:
                if torch.is_tensor(examples[0][key]):
                    collated[key] = torch.stack([e[key] for e in examples], dim=0)
                else:
                    collated[key] = [e[key] for e in examples]
        return collated



