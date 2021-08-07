#!/usr/bin/env python
# coding: utf-8

import json
import random
import tqdm
import os
import copy
import sys
import argparse
QUERY_SMALL_CHANGE_SETS = []

parser = argparse.ArgumentParser()
parser.add_argument("--qra_mode", type=str, help="query rewritten augmentation", default="switch", choices=['delete', 'copy', 'switch',])
parser.add_argument("--task", type=str, help="task selection", default="qa", choices=['qa', 'retrieval'])
args = parser.parse_args()

random.seed(1)
def format_str(string):
    for char in ['\r\n', '\r', '\n']:
        string = string.replace(char, ' ')
    return string
lang = 'python'


def delete_important_words(word_list, replace=''):
    """
        randomly detele an important word in the query or replace (not in QUERY_SMALL_CHANGE_SETS)
    """
    # replace can be [MASK]
    important_word_list = set(word_list) - set(QUERY_SMALL_CHANGE_SETS)
    target = random.sample(important_word_list, 1)[0]
    if replace:
        new_word_list = [item if item!=target else item.replace(target, replace) for item in word_list]
    else:
        new_word_list = [item for item in word_list if item!=target]
    return new_word_list


def copy_important_words(word_list):
    """
        randomly copy an important word in the query to a random place (not in QUERY_SMALL_CHANGE_SETS)
    """
    # replace can be [MASK]
    new_word_list = copy.deepcopy(word_list)
    important_word_list = set(word_list) - set(QUERY_SMALL_CHANGE_SETS)
    target = random.sample(important_word_list, 1)[0]
    position = random.randint(0, len(word_list))
    new_word_list.insert(position, target)
    return new_word_list


def switch_important_words(word_list):
    """
        randomly switch the positions of two words in the query (not in QUERY_SMALL_CHANGE_SETS)
    """
    # replace can be [MASK]
    new_word_list = copy.deepcopy(word_list)
    important_word_list = set(word_list) - set(QUERY_SMALL_CHANGE_SETS)
    target1 = random.sample(important_word_list, 1)[0]
    important_word_list.remove(target1)
    target2 = random.sample(important_word_list, 1)[0]
    idx1, idx2 = word_list.index(target1), word_list.index(target2)
    new_word_list[idx1], new_word_list[idx2] = word_list[idx2], word_list[idx1]
    return new_word_list


if args.task == 'qa':
    file_name = "./qa/cosqa-train.json"
elif args.task == 'retrieval':
    file_name = "./search/cosqa-retrieval-train-19604.json"
with open(file_name, 'r') as fp:
    data = json.load(fp)

codes = set()
code_map = {}
for inst in data:
    # codes.add([inst['code'], inst['code_tokens'], inst['docstring_tokens']])
    codes.add(inst['code'])
    code_map[inst['code']] = [inst['code'], inst['code_tokens'], inst['docstring_tokens']]
print(len(codes))
print(len(code_map))


extended_data_delete = []
extended_data_copy = []
extended_data_switch = []
qra_number = 1
for inst in data:
    if inst['label'] == 0:
        continue
    c = inst['code']

    # delete
    if args.qra_mode == 'delete':
        old_query = inst['doc'].split(' ')
        new_query_list = delete_important_words(copy.deepcopy(old_query))
        new_inst = copy.deepcopy(inst)
        new_inst['idx'] = new_inst['idx'] + '-delete'
        new_inst['doc'] = ' '.join(new_query_list)
        extended_data_delete.append(new_inst)

    # copy
    if args.qra_mode == 'copy':
        old_query = inst['doc'].split(' ')
        new_query_list = copy_important_words(copy.deepcopy(old_query))
        new_inst = copy.deepcopy(inst)
        new_inst['idx'] = new_inst['idx'] + '-copy'
        new_inst['doc'] = ' '.join(new_query_list)
        extended_data_copy.append(new_inst)

    # swtich
    if args.qra_mode == 'switch':
        old_query = inst['doc'].split(' ')
        new_query_list = switch_important_words(copy.deepcopy(old_query))
        new_inst = copy.deepcopy(inst)
        new_inst['idx'] = new_inst['idx'] + '-switch'
        new_inst['doc'] = ' '.join(new_query_list)
        extended_data_switch.append(new_inst)

print("extended data delete num. {} ".format(len(extended_data_delete)))
print("extended data copy num. {} ".format(len(extended_data_copy)))
print("extended data switch num. {} ".format(len(extended_data_switch)))

print("data num. before extension {} ".format(len(data)))
data.extend(extended_data_delete)
data.extend(extended_data_copy)
data.extend(extended_data_switch)
print("data num. after extension {} ".format(len(data)))

save_file_name = file_name.split('.json')[0] + '-qra-{}-{}.json'.format(args.qra_mode, len(data))
print("data save to: {} ".format(save_file_name))
with open(save_file_name, 'w') as fp:
    json.dump(data, fp, indent=1)

