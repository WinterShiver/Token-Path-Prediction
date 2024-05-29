import base64
import copy
import hashlib
import random
from collections import defaultdict, deque
from io import BytesIO
import os
import json
import numpy as np
import requests
import PIL.Image as Image
import torch
import operator
from transformers.tokenization_utils import _is_punctuation

from src.modules.utils import topk_by_partition


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
    if bbox_list is None or bbox_list == []:
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


def is_sentence_piece_subword_v3(token, last_token):
    '''
    v2基础上添加标点符号判断
    '''
    if token is None or last_token is None:
        return False
    if token.startswith('▁'):
        return False
    last_token = last_token.replace('▁', '')
    if len(token) == 1 and _is_punctuation(token):
        return False
    if len(last_token) == 1 and _is_punctuation(last_token):
        return False
    if not is_chinese(token) and not is_chinese(last_token):
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
    return text


def md5(str):
    return hashlib.md5(str.encode('utf-8')).hexdigest()


dis2idx_arr = np.zeros((1000), dtype='int64')
dis2idx_arr[1] = 1
dis2idx_arr[2:] = 2
dis2idx_arr[4:] = 3
dis2idx_arr[8:] = 4
dis2idx_arr[16:] = 5
dis2idx_arr[32:] = 6
dis2idx_arr[64:] = 7
dis2idx_arr[128:] = 8
dis2idx_arr[256:] = 9


def dis2idx(dis):
    dis = min(max(dis, 0), 1000)
    return dis2idx_arr[dis]


def convert_index_to_text(index, type):
    text = "-".join([str(i) for i in index])
    text = text + "-#-{}".format(type)
    return text


def convert_text_to_index(text):
    index, type = text.split("-#-")
    index = [int(x) for x in index.split("-")]
    return index, int(type)


def grid_label_decode(outputs, entities, length):
    class Node:
        def __init__(self):
            self.THW = []  # [(tail, type)]
            self.NNW = defaultdict(set)  # {(head,tail): {next_index}}

    ent_r, ent_p, ent_c = 0, 0, 0
    decode_entities = []

    if outputs.sum() > 10000:
        return ent_c, ent_p, ent_r, decode_entities

    q = deque()
    for instance, ent_set, l in zip(outputs, entities, length):
        predicts = []
        nodes = [Node() for _ in range(l)]
        ent_set = set(ent_set.split(' '))
        for cur in reversed(range(l)):
            heads = []
            for pre in range(cur + 1):
                # THW
                # if instance[cur, pre] > 1:
                #     nodes[pre].THW.append((cur, instance[cur, pre]))
                #     heads.append(pre)
                if instance[cur, pre] == 1:
                    nodes[pre].THW.append((cur, 2))
                    heads.append(pre)
                # NNW
                if pre < cur and instance[pre, cur] == 1:
                    # cur node
                    for head in heads:
                        nodes[pre].NNW[(head, cur)].add(cur)
                    # post nodes
                    for head, tail in nodes[cur].NNW.keys():
                        if tail >= cur and head <= pre:
                            nodes[pre].NNW[(head, tail)].add(cur)
            # entity
            for tail, type_id in nodes[cur].THW:
                if cur == tail:
                    predicts.append(([cur], type_id))
                    continue
                q.clear()
                q.append([cur])
                while len(q) > 0:
                    chains = q.pop()
                    for idx in nodes[chains[-1]].NNW[(cur, tail)]:
                        if idx == tail:
                            predicts.append((chains + [idx], type_id))
                        else:
                            q.append(chains + [idx])

        predicts = set([convert_index_to_text(x[0], x[1]) for x in predicts])
        decode_entities.append([convert_text_to_index(x) for x in predicts])
        ent_r += len(ent_set)
        ent_p += len(predicts)
        ent_c += len(predicts.intersection(ent_set))
    return ent_c, ent_p, ent_r, decode_entities


def cal_f1(c, p, r):
    if r == 0 or p == 0:
        return 0, 0, 0

    r = c / r if r else 0
    p = c / p if p else 0

    if r and p:
        return 2 * p * r / (p + r), p, r
    return 0, p, r


def gb_path_decode(cur_path, linking, scores, result_paths, max_entities=999):
    if len(result_paths) >= max_entities:
        return
    start_idx, start_score = cur_path[-1]
    if start_idx in linking:
        next_idxs = linking[start_idx]
        next_idxs_scores = [scores[start_idx][next_idx] for next_idx in next_idxs]
        # 找最大值
        max_next_idx = np.argmax(next_idxs_scores)
        next_idx = next_idxs[max_next_idx]
        #
        if next_idx in set([idx[0] for idx in cur_path]):
            # 环
            if len(cur_path) == 1:
                result_paths.append([[cur_path[0][0], scores[start_idx][next_idx]]])
            else:
                result_paths.append(copy.copy(cur_path))
        else:
            next_score = scores[start_idx][next_idx]
            cur_path_new = copy.copy(cur_path)
            cur_path_new.append([next_idx, next_score])
            gb_path_decode(cur_path_new, linking, scores, result_paths, max_entities)
    else:
        result_paths.append(copy.copy(cur_path))

def build_token_id_map_from_list(decode_entity_token):
    token_id_set = set()
    token_entity_map = {}
    token_entityid_map = {}
    for entity in decode_entity_token:
        try:
            token_id_list = [int(i) for i in entity.split(":")[-1].split('-') if i != '' and i != ' ']
        except Exception as e:
            print(entity)
            raise e
        for token_id in token_id_list:
            token_entity_map[token_id] = token_id_list
            token_entityid_map[token_id] = entity
        token_id_set = token_id_set.union(token_id_list)
    return token_id_set, token_entity_map, token_entityid_map


def build_token_id_map(decode_entity_token):
    token_id_set = set()
    token_entity_map = {}
    entity_map = {}
    for entity_id, entity in decode_entity_token.items():
        entity_map[entity_id] = entity
        entity_map[entity_id]['group_id'] = -1
        token_id_list = entity['token_ids']
        for token_id in token_id_list:
            token_entity_map[token_id] = entity_id
        token_id_set = token_id_set.union(token_id_list)
    return token_id_set, token_entity_map, entity_map


def build_grouped_result_map(entity_map):
    grouped_result = {
        "分组": []
    }
    group_id2idx_map = {

    }
    group_idx = 0
    for entity_id, entity in entity_map.items():
        group_id = entity['group_id']

        if group_id == -1:
            if entity['label'] not in grouped_result:
                grouped_result[entity['label']] = []
            grouped_result[entity['label']].append({
                "pred": entity['entity'],
                "prob": 1. / (1. + np.exp(-entity['score']))
            })
        else:
            if group_id not in group_id2idx_map:
                group_id2idx_map[group_id] = group_idx
                grouped_result['分组'].append({})
                group_idx += 1
            if entity['label'] not in grouped_result['分组'][group_id2idx_map[group_id]]:
                grouped_result['分组'][group_id2idx_map[group_id]][entity['label']] = []
            grouped_result['分组'][group_id2idx_map[group_id]][entity['label']].append({
                "pred": entity['entity'],
                "prob": 1. / (1. + np.exp(-entity['score']))
            })

    grouped_result_cp = copy.copy(grouped_result)
    for k, v in grouped_result_cp.items():
        if k != '分组':
            sorted_values = sorted(v, key=lambda x: x['prob'], reverse=True)
            grouped_result[k] = sorted_values
        else:
            for group_idx, group in enumerate(v):
                for group_k, group_v in group.items():
                    sorted_values = sorted(group_v, key=lambda x: x['prob'], reverse=True)
                    grouped_result['分组'][group_idx][group_k] = sorted_values

    return grouped_result

def gb_implicit_grouping_decode(entity_idxs, entity_scores, decode_entity_tokens):
    # 并查集解码
    final_groups = []
    for entity_idx, entitiy_score, decode_entity_token in zip(entity_idxs, entity_scores, decode_entity_tokens):
        id_group_map = {}
        group_map = {}
        group_idx = 0
        all_token_id_set, token_entity_map, token_entityid_map = build_token_id_map_from_list(decode_entity_token)

        group_ids = dict()

        def calc_ent_len(entityid):
            return entityid.count('-') + 1

        for start, end in entity_idx:
            if start not in all_token_id_set or end not in all_token_id_set:
                continue

            if token_entityid_map[start] != token_entityid_map[end]:
                key = token_entityid_map[start] + ',' + token_entityid_map[end]
                # key = token_entityid_map[start] + ',' + token_entityid_map[end] \
                #     if token_entityid_map[start] < token_entityid_map[end] \
                #     else token_entityid_map[end] + ',' + token_entityid_map[start]

                if key in group_ids:
                    group_ids[key] += 1. / calc_ent_len(token_entityid_map[start]) / calc_ent_len(token_entityid_map[end])
                else:
                    group_ids[key] = 1. / calc_ent_len(token_entityid_map[start]) / calc_ent_len(token_entityid_map[end])
                if group_ids[key] >= 0.9:
                    group_ids[key] = -5 # 防止同一个分组被多次添加
                    new_group = {start, end}
                    new_group_idx = group_idx
                    group_idx += 1
                    id_group_map[start] = new_group_idx
                    id_group_map[end] = new_group_idx
                    group_map[new_group_idx] = new_group

            # if start not in id_group_map and end not in id_group_map:
            #     # 双节点不在已有分组，创建新分组
            #     new_group = {start, end}
            #     new_group_idx = group_idx
            #     group_idx += 1
            #     id_group_map[start] = new_group_idx
            #     id_group_map[end] = new_group_idx
            #     group_map[new_group_idx] = new_group
            # elif start in id_group_map and end in id_group_map:
            #     # 双节点均已有分组，若分组不同则合并分组
            #     if id_group_map[start] != id_group_map[end]:
            #         a_group_idx = id_group_map[start]
            #         b_group_idx = id_group_map[end]
            #         union_set = group_map[a_group_idx].union(group_map[b_group_idx])
            #         group_map[a_group_idx] = union_set
            #         for idx in group_map[b_group_idx]:
            #             id_group_map[idx] = a_group_idx
            #         del group_map[b_group_idx]
            # else:
            #     # 双节点其中一个已有分组，另一个加入该分组
            #     if start in id_group_map:
            #         inside = start
            #         outside = end
            #     else:
            #         inside = end
            #         outside = start
            #     group_map[id_group_map[inside]].add(outside)
            #     id_group_map[outside] = id_group_map[inside]

        # group_ids = sorted([(group_ids[k], k) for k in group_ids])
        # group_ids.reverse()
        # for g in group_ids:
        #     print(g)
        # input()

        ## 修复漏召回token
        for token_id, group_id in id_group_map.items():
            if token_id in token_entity_map:
                entity_token_ids = token_entity_map[token_id]
                group_map[group_id] = group_map[group_id].union(entity_token_ids)


        label_paths = []
        for group_idx, group_set in group_map.items():
            label_paths.append('-'.join([str(i) for i in sorted(list(group_set))]))
        final_groups.append(label_paths)
    return final_groups


def gb_implicit_grouping_decode_v2(entity_idxs, entity_scores, all_decode_entities):
    # 并查集解码
    final_groups = []
    entity_results = []
    for entity_idx, entitiy_score, decode_entities in zip(entity_idxs, entity_scores, all_decode_entities):
        id_group_map = {}
        group_map = {}
        group_idx = 0
        all_token_id_set, token_entity_map, entity_map = build_token_id_map(decode_entities)

        for start, end in entity_idx:
            if start not in all_token_id_set or end not in all_token_id_set:
                continue

            if start not in id_group_map and end not in id_group_map:
                # 双节点不在已有分组，创建新分组
                new_group = {start, end}
                new_group_idx = group_idx
                group_idx += 1
                id_group_map[start] = new_group_idx
                id_group_map[end] = new_group_idx
                group_map[new_group_idx] = new_group
            elif start in id_group_map and end in id_group_map:
                # 双节点均已有分组，若分组不同则合并分组
                if id_group_map[start] != id_group_map[end]:
                    a_group_idx = id_group_map[start]
                    b_group_idx = id_group_map[end]
                    union_set = group_map[a_group_idx].union(group_map[b_group_idx])
                    group_map[a_group_idx] = union_set
                    for idx in group_map[b_group_idx]:
                        id_group_map[idx] = a_group_idx
                    del group_map[b_group_idx]
            else:
                # 双节点其中一个已有分组，另一个加入该分组
                if start in id_group_map:
                    inside = start
                    outside = end
                else:
                    inside = end
                    outside = start
                group_map[id_group_map[inside]].add(outside)
                id_group_map[outside] = id_group_map[inside]

        for token_id, group_id in id_group_map.items():
            if token_id in token_entity_map:
                entity_id = token_entity_map[token_id]
                entity = entity_map[entity_id]
                entity_map[entity_id]['group_id'] = group_id
                entity_token_ids = entity['token_ids']
                group_map[group_id] = group_map[group_id].union(entity_token_ids)
        result_map = build_grouped_result_map(entity_map)

        label_paths = []
        for group_idx, group_set in group_map.items():
            label_paths.append('-'.join([str(i) for i in sorted(list(group_set))]))
        entity_results.append(result_map)
        final_groups.append(label_paths)
    return final_groups, entity_results

def gb_grouping_decode(entity_idxs, entity_scores):
    # 并查集解码
    final_groups = []
    for entity_idx, entitiy_score in zip(entity_idxs, entity_scores):
        id_group_map = {}
        group_map = {}
        group_idx = 0
        for start, end in entity_idx:
            if start not in id_group_map and end not in id_group_map:
                # 双节点不在已有分组，创建新分组
                new_group = {start, end}
                new_group_idx = group_idx
                group_idx += 1
                id_group_map[start] = new_group_idx
                id_group_map[end] = new_group_idx
                group_map[new_group_idx] = new_group
            elif start in id_group_map and end in id_group_map:
                # 双节点均已有分组，若分组不同则合并分组
                if id_group_map[start] != id_group_map[end]:
                    a_group_idx = id_group_map[start]
                    b_group_idx = id_group_map[end]
                    union_set = group_map[a_group_idx].union(group_map[b_group_idx])
                    group_map[a_group_idx] = union_set
                    for idx in group_map[b_group_idx]:
                        id_group_map[idx] = a_group_idx
                    del group_map[b_group_idx]
            else:
                # 双节点其中一个已有分组，另一个加入该分组
                if start in id_group_map:
                    inside = start
                    outside = end
                else:
                    inside = end
                    outside = start
                group_map[id_group_map[inside]].add(outside)
                id_group_map[outside] = id_group_map[inside]
        label_paths = []
        for group_idx, group_set in group_map.items():
            label_paths.append('-'.join([str(i) for i in sorted(list(group_set))]))
        final_groups.append(label_paths)
    return final_groups

def gb_ner_decode(entities_idxs, entities_scores, max_entities=999):
    # entities_idxs = [[[1, 2], [2, 3], [3, 2], [4, 4], [3, 5]], [[1, 1], [1, 2], [3, 4], [4, 5]]]
    # entities_scores = np.random.random((2, 64, 64))
    # 寻找所有的起始点,没有前置节点的节点，或者只有自己到自己的节点
    ##
    pred_paths = []
    pred_labels = []
    for entities_idx, entities_score in zip(entities_idxs, entities_scores):
        # 每个batch，2*N
        # 建立关系链
        linking = {}  # id ->[ids]
        start_set = set()
        end_set = set()
        for start, end in entities_idx:
            if start not in linking:
                linking[start] = []
            linking[start].append(end)
            start_set.add(start)
            if end != start:
                end_set.add(end)
        # start - end 就是起点
        start_set = start_set - end_set
        # 解析所有路径，以及分数
        result_paths = []
        for start_idx in start_set:
            cur_path = [[start_idx, -1.0]]
            gb_path_decode(cur_path, linking, entities_score, result_paths, max_entities)

        final_paths = []
        final_labels = set([])
        for result_path in result_paths:
            pred_label = '-'.join([str(v[0]) for v in result_path])
            if pred_label not in final_labels:
                final_labels.add(pred_label)
                final_paths.append(result_path)

        pred_paths.append(final_paths)
        pred_labels.append(final_labels)

    return pred_paths, pred_labels

def gb_path_decode_v2(cur_path, linking, linking_reverse, scores, result_paths, max_entities=999):
    if len(result_paths) >= max_entities:
        return
    start_idx, start_score = cur_path[-1]
    if start_idx in linking:
        next_idxs = linking[start_idx]
        next_idxs_scores = [scores[start_idx][next_idx] for next_idx in next_idxs]
        # 找最大值
        max_next_idx = np.argmax(next_idxs_scores)
        next_idx = next_idxs[max_next_idx]

        reverse_start_ids = linking_reverse[next_idx]
        reverse_start_scores = [scores[reverse_start_id][next_idx] for reverse_start_id in reverse_start_ids]
        max_reverse_start_idx = np.argmax(reverse_start_scores)
        reverse_start_id = reverse_start_ids[max_reverse_start_idx]
        if start_idx != reverse_start_id:
            result_paths.append(copy.copy(cur_path))
        else:
            #
            if next_idx in set([idx[0] for idx in cur_path]):
                # 环
                if len(cur_path) == 1:
                    result_paths.append([[cur_path[0][0], scores[start_idx][next_idx]]])
                else:
                    result_paths.append(copy.copy(cur_path))
            else:
                next_score = scores[start_idx][next_idx]
                cur_path_new = copy.copy(cur_path)
                cur_path_new.append([next_idx, next_score])
                gb_path_decode_v2(cur_path_new, linking, linking_reverse, scores, result_paths, max_entities)
    else:
        result_paths.append(copy.copy(cur_path))

def gb_ner_decode_v2(entities_idxs, entities_scores, max_entities=999):
    # entities_idxs = [[[1, 2], [2, 3], [3, 2], [4, 4], [3, 5]], [[1, 1], [1, 2], [3, 4], [4, 5]]]
    # entities_scores = np.random.random((2, 64, 64))
    # 寻找所有的起始点,没有前置节点的节点，或者只有自己到自己的节点
    ##
    pred_paths = []
    pred_labels = []
    for entities_idx, entities_score in zip(entities_idxs, entities_scores):
        # 每个batch，2*N
        # 建立关系链
        linking = {}  # id ->[ids]
        linking_reverse = {}
        start_set = set()
        end_set = set()
        for start, end in entities_idx:
            if start not in linking:
                linking[start] = []
            if end not in linking_reverse:
                linking_reverse[end] = []
            linking[start].append(end)
            linking_reverse[end].append(start)
            start_set.add(start)
            if end != start:
                end_set.add(end)
        # start - end 就是起点
        start_set = start_set - end_set
        # 解析所有路径，以及分数
        result_paths = []
        for start_idx in start_set:
            cur_path = [[start_idx, -1.0]]
            gb_path_decode_v2(cur_path, linking, linking_reverse, entities_score, result_paths, max_entities)

        final_paths = []
        final_labels = set([])
        for result_path in result_paths:
            pred_label = '-'.join([str(v[0]) for v in result_path])
            if pred_label not in final_labels:
                final_labels.add(pred_label)
                final_paths.append(result_path)

        pred_paths.append(final_paths)
        pred_labels.append(final_labels)

    return pred_paths, pred_labels

def nnw_ro_tl_decode(entities_scores, seg_se_maps):
    # entities_idxs = [[[1, 2], [2, 3], [3, 2], [4, 4], [3, 5]], [[1, 1], [1, 2], [3, 4], [4, 5]]]
    # entities_scores = np.random.random((2, 64, 64))
    # 阅读顺序路径，第0个是CLS
    ##
    paths = []
    for entities_score, seg_se_map in zip(entities_scores, seg_se_maps):

        # 获取目标token
        start_token_ids = set()
        end_token_ids = set()
        seg_se_map = json.loads(seg_se_map)
        end2start = dict()
        start2end = dict()
        for k, v in seg_se_map.items():
            start_token_ids.add(v['start_idx'])
            end_token_ids.add(v['end_idx'])
            end2start[v['end_idx']] = v['start_idx']
            start2end[v['start_idx']] = v['end_idx']

        # entities_score = [L,L]
        # greedy search
        # 从0开始
        cur_path = []
        end_id = 0
        end_id_start = end2start[end_id]
        start_token_ids.remove(0)
        end_token_ids.remove(0)
        while len(end_token_ids) > 0:
            next_id_scores = entities_score[end_id]
            next_id = int(np.argmax(next_id_scores))
            max_score = np.max(next_id_scores)
            while (
                    next_id in cur_path
                    or next_id not in start_token_ids
                    or next_id == end_id_start
            ) and max_score > -1000:
                next_id_scores[next_id] = -np.inf
                next_id = int(np.argmax(next_id_scores))
            cur_path.append(next_id)

            start_token_ids.remove(next_id)

            end_id = start2end[next_id]
            end_id_start = end_id
            cur_path.append(end_id)
            end_token_ids.remove(end_id)

        cur_path = [0] + cur_path + [0]

        paths.append(cur_path)
    return paths

def nnw_ro_tl_decode_beam(entities_scores, seg_se_maps, beam_size=8):
    # 手搓阅读顺序版本的beamSearch
    ##
    paths = []
    for entities_score, seg_se_map in zip(entities_scores, seg_se_maps):

        # 获取目标token
        start_token_ids = set()
        end_token_ids = set()
        seg_se_map = json.loads(seg_se_map)
        end2start = dict()
        start2end = dict()
        token_id2seg_id = dict()
        for k, v in seg_se_map.items():
            start_token_ids.add(v['start_idx'])
            end_token_ids.add(v['end_idx'])
            end2start[v['end_idx']] = v['start_idx']
            start2end[v['start_idx']] = v['end_idx']
            token_id2seg_id[v['start_idx']] = int(k)
            token_id2seg_id[v['end_idx']] = int(k)

        # 从0开始
        start_token_ids.remove(0)
        end_token_ids.remove(0)
        start_token_ids = list(start_token_ids)

        # 只选择开始节点的列分数
        next_scores = entities_score[:, start_token_ids]
        # 保留映射关系
        ori_token_id2id_map = dict()
        cur_token_id2id_map = dict()
        for i, s in enumerate(start_token_ids):
            ori_token_id2id_map[s] = i
            cur_token_id2id_map[i] = s

        # [[],score]
        cur_paths = []
        for i in range(len(seg_se_map) - 1):
            # 非重复路径
            start_scores = []
            start_paths = []
            if len(cur_paths) == 0:
                start_scores.append(0.0)
                start_paths.append([0])
            else:
                for path, score in cur_paths:
                    start_paths.append(path)
                    start_scores.append(score)

            # 从start idx开始寻找下一个节点，每个节点找beam_size个候选节点
            # 然后一共会有 len(start_idxs) * beam_size条路径
            # 最终取top beam_size放入路径中
            candidates = []
            for start_path, start_score in zip(start_paths, start_scores):
                # 下一个节点
                start_idx = start_path[-1]
                start_idx = start2end[start_idx]
                next_score = copy.deepcopy(next_scores[start_idx])
                # 下一个节点:1) 不能之前出现过 2）只能是start_token_ids中的节点
                # 将自己和之前出现过的值设置-inf
                for ori_token_id in start_path:
                    if ori_token_id == 0:
                        continue
                    cur_token_id = ori_token_id2id_map[ori_token_id]
                    next_score[cur_token_id] = -np.inf
                # 取top
                if len(next_score) == 1:
                    inds = [0]
                    vals = [next_score[0]]
                else:
                    top_k = max(min(beam_size, len(next_score) - 1), 1)
                    inds, vals = topk_by_partition(next_score,
                                                   k=top_k,
                                                   axis=0,
                                                   ascending=False)
                # 去除-inf的值
                for idx, val in zip(inds, vals):
                    if val == -np.inf:
                        continue
                    next_ori_token_id = cur_token_id2id_map[idx]
                    path_new = copy.deepcopy(start_path)
                    path_new.append(next_ori_token_id)
                    score_new = start_score + val
                    candidates.append([path_new, score_new])
            # 在candidates中选取top beam_size的路径
            candidates = sorted(candidates, key=operator.itemgetter(1), reverse=True)
            cur_paths = candidates[0:beam_size]
        if len(cur_paths) == 0 or len(cur_paths[0]) == 0:
            paths.append([0, 0])
            continue
        cur_path = cur_paths[0][0]
        # 转为segment_id
        seg_path = [0]
        for token_id in cur_path:
            seg_id = token_id2seg_id[token_id]
            if seg_path[-1] != seg_id:
                seg_path.append(seg_id)
        seg_path.append(0)
        paths.append(seg_path)

    return paths

def nnw_ro_decode(entities_scores, ori_lengths):
    # entities_idxs = [[[1, 2], [2, 3], [3, 2], [4, 4], [3, 5]], [[1, 1], [1, 2], [3, 4], [4, 5]]]
    # entities_scores = np.random.random((2, 64, 64))
    # 阅读顺序路径，第0个是CLS
    ##
    paths = []
    for entities_score, ori_length in zip(entities_scores, ori_lengths):
        # entities_score = [L,L]
        # greedy search
        this_path = []
        path_set = set()
        for i in range(0, ori_length):

            start_id = i
            next_scores = entities_score[i][0:ori_length]
            next_id = np.argmax(next_scores)
            max_score = np.max(next_scores)
            while (next_id in path_set or next_id == start_id) and max_score > -1000:
                next_scores[next_id] = -np.inf
                next_id = np.argmax(next_scores)
                max_score = np.max(next_scores)
            this_path.append(next_id)
            path_set.add(next_id)
        this_path = [0] + this_path
        paths.append(this_path)

    return paths

def norm_text(text):
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


if __name__ == '__main__':
    print('')
