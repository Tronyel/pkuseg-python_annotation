# -*- coding:utf-8 -*-
"""
@Time  : 3/14/20 7:41 PM
@Author: Yongbo Wang
@Email : yongbowin@outlook.com
@File  : test.py
@Desc  : 
"""

from collections import Counter
import numpy as np

# words = ['a', 'aa', 'b']
# print(list(map(len, words)))
#
#
words_info = Counter()
words = ['中华人民共和国', '是', '一个', '伟大的', '国家']
# """
# words[:-1] : ['中华人民共和国', '是', '一个', '伟大的']
# words[1:] :  ['是', '一个', '伟大的', '国家']
# """
# for pre, suf in zip(words[:-1], words[1:]):
#     print("{}*{}".format(pre, suf))

#
# def _num_letter_normalize_char(x):
#     return x
#
#
# examples = []
#
# example = [
#     _num_letter_normalize_char(character)  # 判断该 character 是否是数字或者英文字母
#     for word in words  # 对于每一个词 word
#     for character in word  # 对于 word 中的每一个字符 character
# ]
# print(example)
# examples.append(example)
#
# print(examples)


# words_info.update(map(len, words))
#
# print(words_info)


# word = '中华人民共和国'
# word_length = len(word)
# tag_list = []
# for idx, character in enumerate(word):  # 对于每个词中的每个字符 character
#     if word_length == 1:
#         tag = 'B_single'
#         tag_list.append(tag)
#     elif idx == 0:
#         tag = 'B'
#         tag_list.append(tag)
#     elif idx == word_length - 1:
#         tag = 'I_end'
#         tag_list.append(tag)
#     elif idx == 1:
#         tag = 'I_first'
#         tag_list.append(tag)
#     else:
#         tag = 'I'
#         tag_list.append(tag)
#
# # print(word)
# # print(list(word))
# # print(tag_list)

# a = np.random.random(size=(3, 2))

