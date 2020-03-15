# distutils: language = c++
# cython: infer_types=True
# cython: language_level=3
import json
import os
import sys
import pickle
from collections import Counter
from itertools import product

import cython
from pkuseg.config import config

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef get_slice_str(iterable, int start, int length, int all_len):
    """
    包括 当前idx位置的字符的情况：
    (wordary,    idx - l + 1,    l,         length), return "".join(iterable[idx - l + 1: idx + 1])
    (wordary,    idx,            l,         length), return "".join(iterable[idx: idx + l])
    不包括 当前idx位置的字符的情况：
    (wordary,    idx - l,        l,         length), return "".join(iterable[idx -l: idx])
    (wordary,    idx + 1,        l,         length), return "".join(iterable[idx + 1: idx + l +1])
    
    参数：
    (iterable,   int start,      int length, int all_len)
    
    :param iterable: wordary
    :param start: 
    :param length: 
    :param all_len: len(wordary)
    :return: 
    """
    if start < 0 or start >= all_len:
        return ""
    if start + length >= all_len + 1:
        return ""
    return "".join(iterable[start: start + length])

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def __get_node_features_idx(object config not None, int idx, list nodes not None, dict feature_to_idx not None,
                            set unigram not None):
    cdef:
        list flist = []
        Py_ssize_t i = idx
        int length = len(nodes)
        int word_max = config.wordMax
        int word_min = config.wordMin
        int word_range = word_max - word_min + 1


    c = nodes[i]

    # $$ starts feature
    flist.append(0)

    # 8 unigram/bgiram feature
    feat = 'c.' + c
    if feat in feature_to_idx:
        feature = feature_to_idx[feat]
        flist.append(feature)

    if i > 0:
        prev_c = nodes[i - 1]
        feat = 'c-1.' + prev_c
        if feat in feature_to_idx:
            feature = feature_to_idx[feat]
            flist.append(feature)

        feat = 'c-1c.' + prev_c + '.' + c
        if feat in feature_to_idx:
            feature = feature_to_idx[feat]
            flist.append(feature)

    if i + 1 < length:
        next_c = nodes[i + 1]

        feat = 'c1.' + next_c
        if feat in feature_to_idx:
            feature = feature_to_idx[feat]
            flist.append(feature)

        feat = 'cc1.' + c + '.' + next_c
        if feat in feature_to_idx:
            feature = feature_to_idx[feat]
            flist.append(feature)

    if i > 1:
        prepre_char = nodes[i - 2]
        feat = 'c-2.' + prepre_char
        if feat in feature_to_idx:
            feature = feature_to_idx[feat]
            flist.append(feature)

        feat = 'c-2c-1.' + prepre_char + '.' + nodes[i - 1]
        if feat in feature_to_idx:
            feature = feature_to_idx[feat]
            flist.append(feature)

    if i + 2 < length:
        feat = 'c2.' + nodes[i + 2]
        if feat in feature_to_idx:
            feature = feature_to_idx[feat]
            flist.append(feature)

    # no num/letter based features
    if not config.wordFeature:
        return flist

    # 2 * (wordMax-wordMin+1) word features (default: 2*(6-2+1)=10 )
    # the character starts or ends a word

    prelst_in = []
    for l in range(word_max, word_min - 1, -1):
        # length 6 ... 2 (default)
        # "prefix including current c" wordary[n-l+1, n]
        # current character ends word
        tmp = get_slice_str(nodes, i - l + 1, l, length)
        if tmp in unigram:
            feat = 'w-1.' + tmp
            if feat in feature_to_idx:
                feature = feature_to_idx[feat]
                flist.append(feature)

            prelst_in.append(tmp)
        else:
            prelst_in.append("**noWord")

    postlst_in = []
    for l in range(word_max, word_min - 1, -1):
        # "suffix" wordary[n, n+l-1]
        # current character starts word
        tmp = get_slice_str(nodes, i, l, length)
        if tmp in unigram:
            feat = 'w1.' + tmp
            if feat in feature_to_idx:
                feature = feature_to_idx[feat]
                flist.append(feature)

            postlst_in.append(tmp)
        else:
            postlst_in.append("**noWord")

    # these are not in feature list
    prelst_ex = []
    for l in range(word_max, word_min - 1, -1):
        # "prefix excluding current c" wordary[n-l, n-1]
        tmp = get_slice_str(nodes, i - l, l, length)
        if tmp in unigram:
            prelst_ex.append(tmp)
        else:
            prelst_ex.append("**noWord")

    postlst_ex = []
    for l in range(word_max, word_min - 1, -1):
        # "suffix excluding current c" wordary[n+1, n+l]
        tmp = get_slice_str(nodes, i + 1, l, length)
        if tmp in unigram:
            postlst_ex.append(tmp)
        else:
            postlst_ex.append("**noWord")

    # this character is in the middle of a word
    # 2*(wordMax-wordMin+1)^2 (default: 2*(6-2+1)^2=50)

    for pre in prelst_ex:
        for post in postlst_in:
            feat = 'ww.l.' + pre + '*' + post
            if feat in feature_to_idx:
                feature = feature_to_idx[feat]
                flist.append(feature)

    for pre in prelst_in:
        for post in postlst_ex:
            feat = 'ww.r.' + pre + '*' + post
            if feat in feature_to_idx:
                feature = feature_to_idx[feat]
                flist.append(feature)

    return flist


class FeatureExtractor:
    keywords = "-._,|/*:"

    num = set("0123456789." "几二三四五六七八九十千万亿兆零" "１２３４５６７８９０％")
    letter = set(
        "ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ" "ａｂｃｄｅｆｇｈｉｇｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ" "／・－"
    )

    """translate()函数用法举例：
    
    输入：
        from string import maketrans   # 引用 maketrans 函数。
     
        intab = "aeiou"
        outtab = "12345"
        trantab = maketrans(intab, outtab)
    
        str = "this is string example....wow!!!";
        print(str.translate(trantab))
    
    输出：
        th3s 3s str3ng 2x1mpl2....w4w!!!
    """

    # 将 "-._,|/*:" 这8个特殊字符分别转化为 &
    keywords_translate_table = str.maketrans("-._,|/*:", "&&&&&&&&")

    @classmethod
    def keyword_rename(cls, text):
        return text.translate(cls.keywords_translate_table)

    @classmethod
    def _num_letter_normalize_char(cls, character):
        """
        print(set("sw""re"))  # {'r', 'e', 's', 'w'}
        print(set("s  w" "re"))  # {'w', 's', 'r', 'e', ' '}
        print(set("s  w""re"))  # {'w', 's', 'r', 'e', ' '}
        """
        if character in cls.num:
            return "**Num"
        if character in cls.letter:
            return "**Letter"
        return character

    @classmethod
    def normalize_text(cls, text):
        text = cls.keyword_rename(text)
        for character in text:
            if config.numLetterNorm:
                yield cls._num_letter_normalize_char(character)
            else:
                yield character

    def __init__(self):

        self.unigram = set()  # type: Set[str]
        self.bigram = set()  # type: Set[str]
        self.feature_to_idx = {}  # type: Dict[str, int]
        self.tag_to_idx = {}  # type: Dict[str, int]

    def build(self, train_file):
        """处理过程：
        1.逐行读取训练文本，
        2.去除换行符、分隔符
        3.处理数字和英文字母
        4.以字符为单位，以15种方式来抽取特征
        5.定义5中标签
        6.分别将特征和标签转化为id的形式
        """
        with open(train_file, "r", encoding="utf8") as reader:
            lines = reader.readlines()

        examples = []  # type: List[List[List[str]]]

        # first pass to collect unigram and bigram and tag info
        word_length_info = Counter()
        specials = set()
        for line in lines:  # 逐行读取文件
            line = line.strip("\n\r")  # .replace("\t", " ")
            if not line:
                continue

            line = self.keyword_rename(line)  # 特殊字符转化

            """将句子中的分隔符去除 \r \t \n
            >>> '\ra\tnnnb\r\n'.split()
            ['a', 'nnnb']
            
            文本 --> list
            """
            # str.split() without sep sees consecutive whiltespaces as one separator
            # e.g., '\ra \t　b \r\n'.split() = ['a', 'b']
            words = [word for word in line.split()]

            """
            Counter()函数，统计字符串中单个字符的个数
                >>> c = Counter('which')
                >>> c
                Counter({'h': 2, 'w': 1, 'i': 1, 'c': 1})
                >>> 
                >>> c.update('witch')
                >>> c
                Counter({'h': 3, 'w': 2, 'i': 2, 'c': 2, 't': 1})
                >>> 
                >>> d = Counter('watch')
                >>> d
                Counter({'w': 1, 'a': 1, 't': 1, 'c': 1, 'h': 1})
                >>> 
                >>> c.update(d)
                >>> c
                Counter({'h': 4, 'w': 3, 'c': 3, 'i': 2, 't': 2, 'a': 1})
                >>> c['h']
                4
                >>> cc=Counter("象或者另一")
                >>> cc
                Counter({'象': 1, '或': 1, '者': 1, '另': 1, '一': 1})
                >>> cc=Counter("象或  者另一")
                >>> cc
                Counter({' ': 2, '象': 1, '或': 1, '者': 1, '另': 1, '一': 1})
            
            map()函数，在py3返回一个迭代器，在py2返回一个list
                def square(x):  # 计算平方数
                    return x ** 2
                a = map(square, [1, 2, 3, 4, 5])
                print(a, list(a))
                
                输出： <map object at 0x7f5a46454278> [1, 4, 9, 16, 25]
            
            words = ['a', 'aa', 'b']
            print(list(map(len, words)))  # [1, 2, 1]
            
            -------- examples:
            words = ['中华人民共和国', '是', '一个', '伟大的', '国家']
                         len=7      len=1  len=2  len=3    len=2
            words_info.update(map(len, words))
            
            outpues: 
                Counter({2: 2, 7: 1, 1: 1, 3: 1})
            其中 key 是 len， value是长度是这个的有几个
            """
            word_length_info.update(map(len, words))

            """
            set()更新，只保留唯一的元素，将长度大于或等于10的字符串记录下来
            """
            specials.update(word for word in words if len(word) >= 10)

            """
            unigram 也是set()类型，记录所有的字符串，各种长度都有
            """
            self.unigram.update(words)

            """
            words = ['中华人民共和国', '是', '一个', '伟大的', '国家']
            words[:-1] : ['中华人民共和国', '是', '一个', '伟大的']
            words[1:] :  ['是', '一个', '伟大的', '国家']
            
            for pre, suf in zip(words[:-1], words[1:]):
                print("{}*{}".format(pre, suf))
            
            输出：
                中华人民共和国*是
                是*一个
                一个*伟大的
                伟大的*国家
            
            bigram 类型是 set()
            """
            for pre, suf in zip(words[:-1], words[1:]):
                self.bigram.add("{}*{}".format(pre, suf))

            example = [
                self._num_letter_normalize_char(character)  # 判断该 character 是否是数字或者英文字母
                for word in words  # 对于每一个词 word
                for character in word  # 对于 word 中的每一个字符 character
            ]  # type: list[character]
            examples.append(example)  # [[], [], ...]

        max_word_length = max(word_length_info.keys())  # 语料中最长的词的长度
        for length in range(1, max_word_length + 1):
            # 不同于dict，对于Counter而言，找不到key会返回0
            print("length = {} : {}".format(length, word_length_info[length]))
        # print('special words: {}'.format(', '.join(specials)))
        # second pass to get features

        """most_common()函数:
        List the n most common elements and their counts from the most
        common to the least.  If n is None, then list all element counts.

        >>> Counter('abcdeabcdabcaba').most_common(3)
        [('a', 5), ('b', 4), ('c', 3)]
        """
        feature_freq = Counter()

        for example in examples:  # 对于语料中的每一行 line
            for i, _ in enumerate(example):  # 对于每一行中中的每一个 character 字符
                node_features = self.get_node_features(i, example)  # type: list[str]
                feature_freq.update(
                    feature for feature in node_features if feature != "/"
                )

        """
        >>> a=["11", "22", "33", "22"]
        >>> aa=(i for i in a)
        >>> aa
        <generator object <genexpr> at 0x7f24f15c4468>
        >>> type(aa)
        <class 'generator'>
        >>> list(aa)
        ['11', '22', '33', '22']
        """
        feature_set = (
            feature
            for feature, freq in feature_freq.most_common()
            if freq > config.featureTrim  # self.featureTrim = 0, 去掉为0的特征
        )  # 得到一个迭代器对象, 将特征根据数量有多到少降序排列

        # 将特征转化为id
        tot = len(self.feature_to_idx)  # 0, self.feature_to_idx = {}, type: Dict[str, int]
        for feature in feature_set:
            if not feature in self.feature_to_idx:
                self.feature_to_idx[feature] = tot
                tot += 1
        # self.feature_to_idx = {
        #     feature: idx for idx, feature in enumerate(feature_set)
        # }

        # self.nLabel = 5
        if config.nLabel == 2:
            B = B_single = "B"
            I_first = I = I_end = "I"
        elif config.nLabel == 3:
            B = B_single = "B"
            I_first = I = "I"
            I_end = "I_end"
        elif config.nLabel == 4:
            B = "B"
            B_single = "B_single"
            I_first = I = "I"
            I_end = "I_end"
        elif config.nLabel == 5:  # do this
            B = "B"
            B_single = "B_single"
            I_first = "I_first"
            I = "I"
            I_end = "I_end"

        # 5种标签 {"B", "B_single", "I_first", "I", "I_end"}
        tag_set = {B, B_single, I_first, I, I_end}
        # {'B': 0, 'B_single': 1, 'I': 2, 'I_end': 3, 'I_first': 4}
        self.tag_to_idx = {tag: idx for idx, tag in enumerate(sorted(tag_set))}

    def get_node_features_idx(self, idx, nodes):
        return __get_node_features_idx(config, idx, nodes, self.feature_to_idx, self.unigram)

    def get_node_features(self, idx, wordary):
        """
        获取单个character的特征
        :param idx: 当前词的在example列表中的索引
        :param wordary: 即example
        :return: 15个特征具体的值
            特征-1, 开始符号 $$
            特征-2, 当前字符 ("c." + w)
            特征-3, 上一个字符 ("c-1." + wordary[idx - 1])
            特征-4, 下一个字符 ("c1." + wordary[idx + 1])
            特征-5, 上两个字符（不包含上一个）， 一共一个字符 ("c-2." + wordary[idx - 2])
            特征-6, 下两个字符（不包含下一个）， 一共一个字符 ("c2." + wordary[idx + 2])
            特征-7, 二元文法， 当前字符+上一个字符 ("c-1c." + wordary[idx - 1] + "." + w)
            特征-8, 二元文法， 当前字符+下一个字符 ("cc1." + w + "." + wordary[idx + 1])
            特征-9, 二元文法， 上一个字符+上两个位置的字符 --> 一共俩字符 ("c-2c-1." + wordary[idx - 2] + "." + wordary[idx - 1])
            特征-10, 词的长度范围在 [6,5,4,3,2], 如果当前字符作为该词的 [最后一个] 字符，将该词加入到特征中
            特征-11, 词的长度范围在 [6,5,4,3,2], 如果当前字符作为该词的 [开始] 字符，将该词加入到特征中
            特征-12, 词的长度范围在 [6,5,4,3,2], 如果当前字符作为该词的 [最后一个字符的后一个] 字符，将该词加入到特征中
            特征-13, 词的长度范围在 [6,5,4,3,2], 如果当前字符作为该词的 [开始字符的前一个] 字符，将该词加入到特征中
            特征-14, 当前字符作为新词的 [中间字符], 如果 [前边字符串不包含该字符，后边字符串包含该字符]，将该词加入到特征中
            特征-15, 当前字符作为新词的 [中间字符], 如果 [前边字符串包含该字符，后边字符串不包含该字符]，将该词加入到特征中

        在特征抽取的过程中，当前character的标签（5个标签中的一个）也作为一个特征
        """
        cdef int length = len(wordary)  # 当前example长度
        w = wordary[idx]  # 当前词
        flist = []  # 特征列表

        # 1 start feature
        # 特征-1, 开始符号
        flist.append("$$")  # flist = ['$$']

        # 8 unigram/bgiram feature
        # 特征-2, 当前字符
        flist.append("c." + w)  # flist = ['$$', 'c.是']

        # 特征-3, 上一个字符
        if idx > 0:
            flist.append("c-1." + wordary[idx - 1])  # flist = ['$$', 'c.是', 'c-1.是']
        else:
            flist.append("/")

        # 特征-4, 下一个字符
        if idx < len(wordary) - 1:
            flist.append("c1." + wordary[idx + 1])
        else:
            flist.append("/")

        # 特征-5, 上两个字符（不包含上一个）， 一共一个字符
        if idx > 1:
            flist.append("c-2." + wordary[idx - 2])
        else:
            flist.append("/")

        # 特征-6, 下两个字符（不包含下一个）， 一共一个字符
        if idx < len(wordary) - 2:
            flist.append("c2." + wordary[idx + 2])
        else:
            flist.append("/")

        # 特征-7, 二元文法， 当前字符+上一个字符
        if idx > 0:
            flist.append("c-1c." + wordary[idx - 1] + config.delimInFeature + w)  # delimInFeature = "."
        else:
            flist.append("/")

        # 特征-8, 二元文法， 当前字符+下一个字符
        if idx < len(wordary) - 1:
            flist.append("cc1." + w + config.delimInFeature + wordary[idx + 1])
        else:
            flist.append("/")

        # 特征-9, 二元文法， 上一个字符+上两个位置的字符 --> 一共俩字符
        if idx > 1:
            flist.append(
                "c-2c-1."
                + wordary[idx - 2]
                + config.delimInFeature
                + wordary[idx - 1]
            )
        else:
            flist.append("/")

        # 如果不需要word feature 就返回当前flist
        # no num/letter based features
        if not config.wordFeature:  # self.wordFeature = True
            return flist

        # 2 * (wordMax-wordMin+1) word features (default: 2*(6-2+1)=10 )
        # the character starts or ends a word
        # ------------------------- 包含 当前idx位置字符的情况 -------------------------
        tmplst = []
        """
        wordMax = 6, wordMin = 2  字典中词的最大长度是6,最小长度是2
        """
        # 特征-10, 词的长度范围在 [6,5,4,3,2], 如果当前字符作为该词的 [最后一个] 字符，将该词加入到特征中
        for l in range(config.wordMax, config.wordMin - 1, -1):  # for l in range(6, 1, -1) --> [6,5,4,3,2]
            # length 6 ... 2 (default)
            # "prefix including current c" wordary[n-l+1, n+1]
            # current character ends word
            """
            wordary 为list， 元素为单个character
            get_slice_str() 函数返回 iterable[start: start + length]
            即， wordary[idx - l + 1 : idx + 1]， 并且 l in [6,5,4,3,2]
            
            例子：
            如果l=6, 结果如下： 范围是 wordary[idx - 5, idx+1]， idx位置为 '国'
            
            ['中', '华', '人', '民', '共', '和', '国', '今', '天', '成', '立', '了', '！']
              -6   -5    -4    -3   -2    -1   idx    1    2     3    4     5    6
            
            得出结果为 '华人民共和国'， 长度为6
            """
            tmp = get_slice_str(wordary, idx - l + 1, l, length)  # [idx -l +1 : idx + 1]
            if tmp != "":
                if tmp in self.unigram:  # self.unigram中存储有所有的词，如果在其中，认为该新词 '华人民共和国' 是存在的
                    flist.append("w-1." + tmp)  # 词长度=6时，当前字符为 '国', 此时 tmp = '华人民共和国'
                    tmplst.append(tmp)
                else:  # 认为该新词 '华人民共和国' 不存在
                    flist.append("/")
                    tmplst.append("**noWord")
            else:
                flist.append("/")
                tmplst.append("**noWord")
        prelst_in = tmplst  # 当前字符在新词中，以该字符为结尾的长度在 [6,5,4,3,2] 的字符串

        tmplst = []
        # 特征-11, 词的长度范围在 [6,5,4,3,2], 如果当前字符作为该词的 [开始] 字符，将该词加入到特征中
        for l in range(config.wordMax, config.wordMin - 1, -1):  # [6,5,4,3,2]
            # "suffix" wordary[n, n+l-1]
            # current character starts word
            """例子：
            如果l=6, 结果如下： 范围是 wordary[idx : idx+6]， idx位置为 '国'
            
            ['中', '华', '人', '民', '共', '和', '国', '今', '天', '成', '立', '了', '！']
              -6   -5    -4    -3   -2    -1   idx    1    2     3    4     5    6
            
            得出结果为 '国今天成立了'， 长度为6
            """
            tmp = get_slice_str(wordary, idx, l, length)  # [idx : idx + l]
            if tmp != "":
                if tmp in self.unigram:
                    flist.append("w1." + tmp)
                    tmplst.append(tmp)
                else:
                    flist.append("/")
                    tmplst.append("**noWord")
            else:
                flist.append("/")
                tmplst.append("**noWord")
        postlst_in = tmplst

        # ------------------------- 不包含 当前idx位置字符的情况 -----------------------
        # these are not in feature list
        tmplst = []
        # 特征-12, 词的长度范围在 [6,5,4,3,2], 如果当前字符作为该词的 [最后一个字符的后一个] 字符，将该词加入到特征中
        for l in range(config.wordMax, config.wordMin - 1, -1):  # [6,5,4,3,2]
            # "prefix excluding current c" wordary[n-l, n-1]
            """例子：
            如果l=6, 结果如下： 范围是 wordary[idx - 6 : idx]， idx位置为 '国'
            
            ['中', '华', '人', '民', '共', '和', '国', '今', '天', '成', '立', '了', '！']
              -6   -5    -4    -3   -2    -1   idx    1    2     3    4     5    6
            
            得出结果为 '中华人民共和'， 长度为6
            """
            tmp = get_slice_str(wordary, idx - l, l, length)  # [idx -l : idx]
            if tmp != "":
                if tmp in self.unigram:
                    tmplst.append(tmp)
                else:
                    tmplst.append("**noWord")
            else:
                tmplst.append("**noWord")
        prelst_ex = tmplst

        tmplst = []
        # 特征-13, 词的长度范围在 [6,5,4,3,2], 如果当前字符作为该词的 [开始字符的前一个] 字符，将该词加入到特征中
        for l in range(config.wordMax, config.wordMin - 1, -1):  # [6,5,4,3,2]
            # "suffix excluding current c" wordary[n+1, n+l]
            """例子：
            如果l=6, 结果如下： 范围是 wordary[idx + 1 : idx + 7]， idx位置为 '国'
            
            ['中', '华', '人', '民', '共', '和', '国', '今', '天', '成', '立', '了', '！']
              -6   -5    -4    -3   -2    -1   idx    1    2     3    4     5    6
            
            得出结果为 '今天成立了！'， 长度为6
            """
            tmp = get_slice_str(wordary, idx + 1, l, length)  # [idx + 1 : idx + l + 1]
            if tmp != "":
                if tmp in self.unigram:
                    tmplst.append(tmp)
                else:
                    tmplst.append("**noWord")
            else:
                tmplst.append("**noWord")
        postlst_ex = tmplst

        # ------------------------- 当前idx位置字符 位于中间的情况 ----------------------
        # this character is in the middle of a word
        # 2*(wordMax-wordMin+1)^2 (default: 2*(6-2+1)^2=50)

        # 特征-14, 当前字符作为新词的 [中间字符], 如果 [前边字符串不包含该字符，后边字符串包含该字符]，将该词加入到特征中
        """
        ['中', '华', '人', '民', '共', '和', '国', '今', '天', '成', '立', '了', '！']
          -6   -5    -4    -3   -2    -1   idx    1    2     3    4     5    6
         --------------------------------  -------------------------------
               --------------------------  --------------------------
                     --------------------  --------------------
                          ---------------  ---------------
                                ---------  ---------
                                     ----  ----
        """
        for pre in prelst_ex:  # 当前字符前边字符组合，不包含当前字符
            for post in postlst_in:  # 当前字符后边字符组合，包含当前字符
                bigram = pre + "*" + post  # 词的二元文法
                if bigram in self.bigram:
                    flist.append("ww.l." + bigram)
                else:
                    flist.append("/")

        # 特征-15, 当前字符作为新词的 [中间字符], 如果 [前边字符串包含该字符，后边字符串不包含该字符]，将该词加入到特征中
        """
        ['中', '华', '人', '民', '共', '和', '国', '今', '天', '成', '立', '了', '！']
          -6   -5    -4    -3   -2    -1   idx    1    2     3    4     5    6
               --------------------------------  -------------------------------
                     --------------------------  --------------------------
                           --------------------  --------------------
                                ---------------  ---------------
                                      ---------  ---------
                                           ----  ----
        """
        for pre in prelst_in:  # 当前字符前边字符组合，包含当前字符
            for post in postlst_ex:  # 当前字符后边字符组合，不包含当前字符
                bigram = pre + "*" + post
                if bigram in self.bigram:
                    flist.append("ww.r." + bigram)
                else:
                    flist.append("/")

        return flist

    def convert_feature_file_to_idx_file(self, feature_file, feature_idx_file, tag_idx_file):
        """
        将特征文件中特征转化为id
        For training : ("xxx/train.feat.txt", "xxx/ftrain.txt", "xxx/gtrain.txt")
        For test     : ("xxx/test.feat.txt", "xxx/ftest.txt", "xxx/gtest.txt")
        :param feature_file:
        :param feature_idx_file:
        :param tag_idx_file:
        :return:
        """

        with open(feature_file, "r", encoding="utf8") as reader:
            lines = reader.readlines()

        with open(feature_idx_file, "w", encoding="utf8") as f_writer, \
                open(tag_idx_file, "w", encoding="utf8") as t_writer:

            # 文件的首行写入字典的大小，空一行
            f_writer.write("{}\n\n".format(len(self.feature_to_idx)))
            t_writer.write("{}\n\n".format(len(self.tag_to_idx)))

            tags_idx = []  # type: List[str]
            features_idx = []  # type: List[List[str]]
            for line in lines:  # 特征文件中的每一行
                line = line.strip()

                # 代码，先不看循环，先看循环下边的执行语句
                # 每一行（或者每个句子）结束之后，开始执行下边循环，因为不同行之间的字符特征字符串使用一个单独的空行隔开了
                if not line:  # 句子结束，开始执行循环中的代码
                    # sentence finish
                    for feature_idx in features_idx:  # 对于一句话中的单个字符的特征id集合而言
                        if not feature_idx:
                            f_writer.write("0\n")
                        else:
                            f_writer.write(",".join(map(str, feature_idx)))  # 各个特征之间的id使用','进行分割拼接，作为一行
                            f_writer.write("\n")
                    f_writer.write("\n")  # 每句话的特征id字符串和标签id之间使用空行进行分割

                    """只有一个元素时，','不会加入
                    >>> a=["B"]
                    >>> ",".join(a)
                    'B'
                    """
                    t_writer.write(",".join(map(str, tags_idx)))
                    t_writer.write("\n\n")  # 每两句话之间依然使用空行进行分割

                    tags_idx = []
                    features_idx = []
                    continue

                # 每一行（或者每个句子）结束之前，执行下边语句
                splits = line.split(" ")  # 将每个characher的特征字符串进行分割，分割出的特征值的集合
                feature_idx = [
                    self.feature_to_idx[feat]
                    for feat in splits[:-1]  # 因为最后一个是标签，需要单独进行id转换
                    if feat in self.feature_to_idx
                ]
                """
                features_idx : 
                    [[xx, xx, ...],        [],            [],        ...]
                    字符1的特征id集合  字符2的特征id集合  字符3的特征id集合  ...
                
                tags_idx : 
                    [xx,             xx,           xx,     ...]
                    字符1的标签id  字符2的标签id  字符3的标签id  ...
                """
                features_idx.append(feature_idx)
                tags_idx.append(self.tag_to_idx[splits[-1]])

    def convert_text_file_to_feature_file(self, text_file, conll_file=None, feature_file=None):
        """
        将文本文件转为特征文件
        For training : ("xxx/data/small_training.utf8", "xxx/train.conll.txt", "xxx/train.feat.txt")
        For test     : ("xxx/data/small_test.utf8", "xxx/test.conll.txt", "xxx/test.feat.txt")
        :param text_file:
        :param conll_file:  将每个字符的标签写入到该文件中，每一行一个字符，格式为 '国 B'，行与行之间的字符使用一个空行进行分割
        :param feature_file:  将当前字符的特征字符串和标签组成的list，使用空格进行拼接，每个字符占一行，行与行之间的字符使用一个空行进行分割
        """

        """
        >>> "{}.conll{}".format(*os.path.split("/home/wyb/data/train.json"))
        '/home/wyb/data.conlltrain.json'
        >>> os.path.split("/home/wyb/data/train.json")
        ('/home/wyb/data', 'train.json')
        """
        # 定义文件名
        if conll_file is None:
            conll_file = "{}.conll{}".format(*os.path.split(text_file))
        if feature_file is None:
            feature_file = "{}.feat{}".format(*os.path.split(text_file))

        # self.nLabel = 5
        if config.nLabel == 2:
            B = B_single = "B"
            I_first = I = I_end = "I"
        elif config.nLabel == 3:
            B = B_single = "B"
            I_first = I = "I"
            I_end = "I_end"
        elif config.nLabel == 4:
            B = "B"
            B_single = "B_single"
            I_first = I = "I"
            I_end = "I_end"
        elif config.nLabel == 5:  # do this
            B = "B"
            B_single = "B_single"  # 单个字符独立成词
            I_first = "I_first"
            I = "I"
            I_end = "I_end"

        conll_line_format = "{} {}\n"

        with open(text_file, "r", encoding="utf8") as reader, \
                open(conll_file, "w", encoding="utf8") as c_writer, \
                open(feature_file, "w", encoding="utf8") as f_writer:
            for line in reader:  # 逐行读取原始文本文件 line
                line = line.strip()
                if not line:
                    continue
                # 将包含在 '-._,|/*:&' 中的字符统一替换为 '&'， 输入文件词之间是以空格隔开的，将其分隔
                words = self.keyword_rename(line).split()
                example = []  # 存储一行中所有的字符
                tags = []     # 存储一行中每个字符对应的标签
                for word in words:  # 对于每行中的每个词 word
                    word_length = len(word)
                    """
                    将词中的每个字符打上标签 : 
                    'B_single' : 单个字符, 独立成词
                    'B'        : 多个字符, 开始的字符
                    'I_end'    : 多个字符, 结尾的字符
                    'I_first'  : 多个字符, 第二个字符 (一般文本字符串开始和结尾都会有特殊标记来标注一句话的开始和结束)
                    'I'        : 多个字符, 位于中间的字符
                    
                    eg, word = '中华人民共和国'
                        ['中', '华',       '人', '民', '共', '和', '国']
                        ['B',  'I_first', 'I',  'I',  'I', 'I',  'I_end']
                    """
                    for idx, character in enumerate(word):  # 对于每个词中的每个字符 character
                        if word_length == 1:
                            tag = B_single
                        elif idx == 0:
                            tag = B
                        elif idx == word_length - 1:
                            tag = I_end
                        elif idx == 1:
                            tag = I_first
                        else:
                            tag = I
                        c_writer.write(conll_line_format.format(character, tag))  # 写入 conll_file 文件，每个字符占一行

                        if config.numLetterNorm:  # self.numLetterNorm = True
                            example.append(  # example=[] 的范围是在本行内 line
                                # 判断当前字符是否是 英文字母或者数字， 如果是，返回特殊标记字符串，否则返回字符本身
                                self._num_letter_normalize_char(character)
                            )
                        else:
                            example.append(character)
                        tags.append(tag)  # tags=[] 的范围是在本行内 line
                c_writer.write("\n")  # 每行处理完之后，使用空行隔开

                for idx, tag in enumerate(tags):  # 对于一行文本来说 line
                    features = self.get_node_features(idx, example)
                    features = [
                        (feature if feature in self.feature_to_idx else "/")  # self.feature_to_idx 是从训练语料中获取的
                        for feature in features
                    ]
                    features.append(tag)  # 当前character的标签也作为一个特征
                    """
                    写入到 feature_file 文件中， 每一行是所抽取的当前字符的(15+1)类特征字符串的拼接，使用空格进行拼接
                    """
                    f_writer.write(" ".join(features))
                    f_writer.write("\n")
                f_writer.write("\n")  # 每行处理完之后，使用空行隔开

    def save(self, model_dir=None):
        """
        "xxx/models/ctb8/features.pkl"
        """
        if model_dir is None:
            model_dir = config.modelDir  # "xxx/models/ctb8"
        data = {}
        data["unigram"] = sorted(list(self.unigram))  # sorted() 默认是升序排列
        data["bigram"] = sorted(list(self.bigram))
        data["feature_to_idx"] = self.feature_to_idx
        data["tag_to_idx"] = self.tag_to_idx

        with open(os.path.join(model_dir, 'features.pkl'), 'wb') as writer:
            pickle.dump(data, writer, protocol=pickle.HIGHEST_PROTOCOL)

        # with open(
        #     os.path.join(config.modelDir, "features.json"), "w", encoding="utf8"
        # ) as writer:
        #     json.dump(data, writer, ensure_ascii=False)

    @classmethod
    def load(cls, model_dir=None):
        if model_dir is None:
            model_dir = config.modelDir
        extractor = cls.__new__(cls)  # TODO

        feature_path = os.path.join(model_dir, "features.pkl")
        if os.path.exists(feature_path):
            with open(feature_path, "rb") as reader:
                data = pickle.load(reader)
            extractor.unigram = set(data["unigram"])
            extractor.bigram = set(data["bigram"])
            extractor.feature_to_idx = data["feature_to_idx"]
            extractor.tag_to_idx = data["tag_to_idx"]

            return extractor

        print(
            "WARNING: features.pkl does not exist, try loading features.json",
            file=sys.stderr,
        )

        feature_path = os.path.join(model_dir, "features.json")
        if os.path.exists(feature_path):
            with open(feature_path, "r", encoding="utf8") as reader:
                data = json.load(reader)
            extractor.unigram = set(data["unigram"])
            extractor.bigram = set(data["bigram"])
            extractor.feature_to_idx = data["feature_to_idx"]
            extractor.tag_to_idx = data["tag_to_idx"]
            extractor.save(model_dir)
            return extractor
        print(
            "WARNING: features.json does not exist, try loading using old format",
            file=sys.stderr,
        )

        with open(
                os.path.join(model_dir, "unigram_word.txt"),
                "r",
                encoding="utf8",
        ) as reader:
            extractor.unigram = set([line.strip() for line in reader])

        with open(
                os.path.join(model_dir, "bigram_word.txt"),
                "r",
                encoding="utf8",
        ) as reader:
            extractor.bigram = set(line.strip() for line in reader)

        extractor.feature_to_idx = {}
        feature_base_name = os.path.join(model_dir, "featureIndex.txt")
        for i in range(10):
            with open(
                    "{}_{}".format(feature_base_name, i), "r", encoding="utf8"
            ) as reader:
                for line in reader:
                    feature, index = line.split(" ")
                    feature = ".".join(feature.split(".")[1:])
                    extractor.feature_to_idx[feature] = int(index)

        extractor.tag_to_idx = {}
        with open(
                os.path.join(model_dir, "tagIndex.txt"), "r", encoding="utf8"
        ) as reader:
            for line in reader:
                tag, index = line.split(" ")
                extractor.tag_to_idx[tag] = int(index)

        print(
            "INFO: features.json is saved",
            file=sys.stderr,
        )
        extractor.save(model_dir)

        return extractor
