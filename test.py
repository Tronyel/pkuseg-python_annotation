# -*- coding:utf-8 -*-
"""
@Time  : 3/14/20 7:41 PM
@Author: Yongbo Wang
@Email : yongbowin@outlook.com
@File  : test.py
@Desc  : 
"""

# words = ['a', 'aa', 'b']
# print(list(map(len, words)))
#
#
words = ['中华人民共和国', '是', '一个', '伟大的', '国家']
# """
# words[:-1] : ['中华人民共和国', '是', '一个', '伟大的']
# words[1:] :  ['是', '一个', '伟大的', '国家']
# """
# for pre, suf in zip(words[:-1], words[1:]):
#     print("{}*{}".format(pre, suf))


def _num_letter_normalize_char(x):
    return x


examples = []

example = [
    _num_letter_normalize_char(character)  # 判断该 character 是否是数字或者英文字母
    for word in words  # 对于每一个词 word
    for character in word  # 对于 word 中的每一个字符 character
]
print(example)
examples.append(example)

print(examples)