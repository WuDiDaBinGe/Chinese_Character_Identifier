import jieba
from itertools import permutations
# 结巴分词 识别语序
def recog_order_jieba(str):
    l = len(str)  # l表示输入字符串个数
    word_list = _permutation(str)  # 获得该字符串的所有排列方式
    possible_words = []  # 用来存放语序可能正确的词
    for word in word_list:  # 编列所有排列方式
        seg_list = jieba.lcut(word, cut_all=True)  # 对某一种排列方式使用结巴分词
        index = find_longest(seg_list)  # 寻找结巴分词返回的列表中字符串最长的索引，并返回
        if len(seg_list[index]) == l:  # 若最长的字符串与输入的字符串长度相同，则加入可能正确列表
            possible_words.append(seg_list[index])
    if len(possible_words) == 1:  # 遍历完后，若可能正确的列表只有一个元素，那么他就是正确的，返回
        return possible_words[0]
    elif len(possible_words) > 1:  # 若有可能正确列表中若有多个元素，则选取词频高的返回
        return highest_frequency(possible_words)
    else:  # 如果可能正确的列表元素为0，则返回0
        return 0

# 获得汉字的所有排列方式
def _permutation(str, r=None):
    word_list = list(permutations(str, r))
    for i in range(len(word_list)):
        word_list[i] = ''.join(word_list[i])
    return word_list


# 寻找列表中最长的词
def find_longest(list):
    l = 0
    index = 0
    for i, word in enumerate(list):
        if len(word) > l:
            l = len(word)
            index = i
    return index


# 输入词列表，返回结巴分词内词频最高的词
def highest_frequency(possible_words):
    word_dict = file2dict('dict.txt')
    possible_dict = {}
    for possible_word in possible_words:
        possible_dict[word_dict[possible_word]] = possible_word
    sortedList = sortedDictValues(possible_dict)
    print(sortedList)
    return sortedList[-1][1]

# 对输入的字典根据key大小排序
def sortedDictValues(di):
    return [(k, di[k]) for k in sorted(di.keys())]

# 将文件数据转换为字典
def file2dict(filename):
    with open(filename,encoding='UTF-8') as f:
        array_lines = f.readlines()
    returnDict = {}
    # 以下三行解析文件数据到列表
    for line in array_lines:
        line = line.strip()
        listFromLine = line.split()
        returnDict[listFromLine[0]] = int(listFromLine[1])
    return returnDict

if __name__ == '__main__':
    str=recog_order_jieba("命期生周")
    print(str)