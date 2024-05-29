import base64
import hashlib
import random
from dataclasses import dataclass
from io import BytesIO
import os
from typing import Optional, Tuple

import numpy as np
import requests
import PIL.Image as Image
import torch
from transformers.utils import ModelOutput


def labels2bio(labels, default_label):
    '''labels2bio
    '''
    label_bios = []

    pre_label = None

    for idx, label in enumerate(labels):

        if label == default_label:
            label_bios.append(label)
        elif label == pre_label:
            label_bios.append('I-{}'.format(label))
        else:
            label_bios.append('B-{}'.format(label))

        pre_label = label

    return label_bios


def box_norm(box, width, height, norm_width, norm_height):
    def clip(min_num, num, max_num):
        return min(max(num, min_num), max_num)

    x0, y0, x1, y1 = box

    # ## 四角坐标兼容
    # if isinstance(x0, list):
    #     y0 = x0[1]
    #     x0 = x0[0]
    #     y1 = x1[1]
    #     x1 = x1[0]

    x0 = clip(0, int((x0 / width) * norm_width), norm_width)
    y0 = clip(0, int((y0 / height) * norm_height), norm_height)
    x1 = clip(0, int((x1 / width) * norm_width), norm_width)
    y1 = clip(0, int((y1 / height) * norm_height), norm_height)
    return [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]


def points4to2(box):
    '''
    将4点坐标转为2点坐标
    '''
    if box is None or len(box) == 0 or not isinstance(box[0], list):
        return box

    xs, ys = list(zip(*box))
    box = [min(xs), min(ys), max(xs), max(ys)]
    return box


def merge_bbox(bbox_list):
    if bbox_list is None:
        return None
    x0, y0, x1, y1 = list(zip(*bbox_list))
    return [min(x0), min(y0), max(x1), max(y1)]


def get_pil_image(image):
    if image.startswith("http"):
        response = requests.get(image)
        image_pil = Image.open(BytesIO(response.content))
    elif os.path.exists(image):
        image_pil = Image.open(image)
    else:
        image_pil = base64_pil(image)
    return image_pil.convert('RGB')


def base64_pil(base64str):
    image = base64.b64decode(base64str)
    image = BytesIO(image)
    image = Image.open(image)
    return image


def _is_chinese_char(cp):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    if (
            (cp >= 0x4E00 and cp <= 0x9FFF)
            or (cp >= 0x3400 and cp <= 0x4DBF)  #
            or (cp >= 0x20000 and cp <= 0x2A6DF)  #
            or (cp >= 0x2A700 and cp <= 0x2B73F)  #
            or (cp >= 0x2B740 and cp <= 0x2B81F)  #
            or (cp >= 0x2B820 and cp <= 0x2CEAF)  #
            or (cp >= 0xF900 and cp <= 0xFAFF)
            or (cp >= 0x2F800 and cp <= 0x2FA1F)  #
    ):  #
        return True

    return False


def is_chinese(word: str):
    if word is None:
        return 0
    for char in word:
        char = ord(char)
        if not _is_chinese_char(char):
            return 0
    return 1


def add_sub_symbol(bert_tokens, chinese_word_set):
    if not chinese_word_set:
        return bert_tokens
    max_word_len = max([len(w) for w in chinese_word_set])

    bert_word = bert_tokens
    start, end = 0, len(bert_word)
    while start < end:
        single_word = True
        if is_chinese(bert_word[start]):
            l = min(end - start, max_word_len)
            for i in range(l, 1, -1):
                whole_word = "".join(bert_word[start: start + i])
                if whole_word in chinese_word_set:
                    for j in range(start + 1, start + i):
                        bert_word[j] = "##" + bert_word[j]
                    start = start + i
                    single_word = False
                    break
        if single_word:
            start += 1
    return bert_word


def get_chinese_word(tokens):
    word_set = set()

    for token in tokens:
        chinese_word = len(token) > 1 and is_chinese(token)
        if chinese_word:
            word_set.add(token)
    word_list = list(word_set)
    return word_list


def is_sentence_piece_subword(token):
    if token is None:
        return False

    if not is_chinese(token) and not token.startswith("▁"):
        return True

    return False


def is_sentence_piece_subword_v2(token, last_token):
    '''
    example：中文askhdjdh数字12345空格 askhdjdh
    Tokenizer后：['▁', '中文', 'as', 'kh', 'd', 'j', 'dh', '数字', '12', '345', '空', '格', '▁ask', 'hd', 'j', 'dh']
    wordids:[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]

    所以无法通过wordids判断wordpiece，"中文+英文，中文+数字，中文+标点" 如果没有空格分隔，则没有_，需要特殊处理

    '''
    if token is None:
        return False

    if not is_chinese(token) and not token.startswith("▁") and not is_chinese(last_token):
        return True

    return False


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def norm_text_v0(text):
    text = text.replace('℃', 'C')
    text = text.replace('℉', 'F')
    text = text.replace('㎡', 'm')
    text = text.replace('ⅱ', 'i')
    text = text.replace('⒈', '1')
    text = text.replace('⒉', '2')
    text = text.replace('⒊', '3')
    text = text.replace('⒋', '4')
    text = text.replace('⒌', '5')
    text = text.replace('⒍', '6')
    text = text.replace('⒎', '7')
    text = text.replace('⒏', '8')
    text = text.replace('⒐', '9')
    text = text.replace('：', ':')
    text = text.replace('；', ';')
    text = text.replace('，', ',')
    text = text.replace('（', '(')
    text = text.replace('）', ')')
    text = text.replace('？', '?')
    return text


def md5(str):
    return hashlib.md5(str.encode('utf-8')).hexdigest()


def get_top_k(a, k):
    return sorted(range(len(a)), key=lambda i: a[i], reverse=True)[:k]


@dataclass
class BaseModelOutputWithRel2DPos(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    rel_2d_pos: torch.FloatTensor = None


def distance(p0, p1):
    x1, y1 = p0
    x2, y2 = p1
    distance = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
    return distance


def topk_by_partition(input, k, axis=None, ascending=True):
    if not ascending:
        input *= -1
    ind = np.argpartition(input, k, axis=axis)
    ind = np.take(ind, np.arange(k), axis=axis)  # k non-sorted indices
    input = np.take_along_axis(input, ind, axis=axis)  # k non-sorted values

    # sort within k elements
    ind_part = np.argsort(input, axis=axis)
    ind = np.take_along_axis(ind, ind_part, axis=axis)
    if not ascending:
        input *= -1
    val = np.take_along_axis(input, ind_part, axis=axis)
    return ind, val


if __name__ == '__main__':
    input = np.asarray([1, 2, 3, 4])
    rs = topk_by_partition(input, k=2, axis=0,ascending=False)
    print(rs)
