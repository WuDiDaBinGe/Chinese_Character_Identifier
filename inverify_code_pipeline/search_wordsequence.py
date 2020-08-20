import requests
from lxml import etree
from inverify_code_pipeline.jieba_wordsequence_API import _permutation
import threading
# 搜索引擎搜索关键字,返回相关列表
def search_engine(word):
    headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.139 Safari/537.36'
    }
    r = requests.get('https://www.baidu.com/s?wd=' + word, headers=headers)
    html = etree.HTML(r.text)
    related_words1 = html.xpath('//*[@id="rs"]/table//tr//th/a/text()')
    related_words2 = html.xpath('//div[@id="content_left"]//a//em/text()')
    related_words = related_words1 + related_words2
    return related_words


# 调用一次线程，每一个线程对输入字符串进行百度搜索，返回相关词的列表
def search(word):
    related_words = search_engine(word)
    global all_related
    all_related = all_related + related_words


# 通过搜索引擎识别语序
def search_engine_recog(str):
    word_list = _permutation(str)  # 获得排列
    global flags
    flags = [0] * len(word_list)  # 标志位
    threads = []

    global all_related  # 记录所有排列组合进行百度搜索后返回的列表
    all_related=[]
    for word in word_list:  # 遍历所有可能的排列组合，进行百度搜索
        thread = threading.Thread(target=search, args=[word])
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()
    print(all_related)
    for i, word in enumerate(word_list):  # 遍历所有排列
        flag = 0
        for related_word in all_related:  # 对每一个排列统计在所有相关词语列表中出现的次数
            if word in related_word:
                flag = flag + 1
        flags[i] = flag
    all_related = []  # 清空
    index = flags.index(max(flags))  # 找到标志位最大的索引
    return word_list[index]

if __name__ == '__main__':
    str_=search_engine_recog("命期生周")
    print(str_)