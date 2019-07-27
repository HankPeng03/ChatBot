import os
import jieba
from langconv import Converter
from gensim.models import Word2Vec
import json

def Traditional2Simplified(sentence):
    '''
    将sentence中的繁体字转为简体字
    :param sentence: 待转换的句子
    :return: 将句子中繁体字转换为简体字之后的句子
    '''
    sentence = Converter('zh-hans').convert(sentence)
    return sentence

if __name__ == '__main__':

    # 读取数据
    corpus_path = "Data/raw_chat_corpus/ptt-42w/Gossiping-QA-Dataset.txt"
    with open(corpus_path, encoding="utf8") as f:
            corpus = f.readlines()
    corpus = corpus[:1000]  # 取Top1000个，减少数据处理量
    corpus = [str.split(i, '\t') for i in corpus]  # 划分为Q与A
    corpus = [[Traditional2Simplified(str.strip(j)) for j in i] for i in corpus]  # 去除\n
    # 将繁体转换为简体
    print("语料库读取完成".center(30, "="))

    # 分词并构建词典
    # jieba.load_userdict("")
    corpus_cut = [[jieba.lcut(j) for j in i] for i in corpus]
    print("分词完成".center(30, "="))

    # 构建词典单词与数值id的映射关系
    all_dict = []
    for i in corpus_cut:
        for j in i:
            all_dict.extend(j)
    _UNK, _BOS, _EOS, _PAD = "_UNK", "_BOS", "_EOS", "_PAD"
    all_dict = list(set(all_dict)) + [_UNK, _BOS, _EOS, _PAD]  # 添加特殊符号
    print("词典构建完成".center(30, "="))
    id2word = {i: w for i, w in enumerate(all_dict)}
    word2id = {w: i for i, w in enumerate(all_dict)}
    print("单词与数值id的映射关系构建完成".center(30, "="))

    # 将语料转为id向量
    ids = [[[word2id.get(j, word2id[_UNK]) for j in i] for i in w] for w in corpus_cut]

    # 拆分为source、target
    fromids = [w[0] for w in ids]
    toids = [w[1] for w in ids]

    # 词向量训练
    model_path = "Temp/word2vec.model"
    if not os.path.exists(model_path):
        emb_size = 100
        tmp = [list(map(str, i)) for i in fromids] + [list(map(str, j)) for j in toids]
        model = Word2Vec(tmp, size=emb_size, window=10, min_count=1, workers=4)
        model.save(model_path)
    else:
        print("加载已经训练好的词向量模型")
        model = Word2Vec.load(model_path)

    # 保存文件
    with open("Temp/data.json", 'w') as f:
        json.dump({"fromids":fromids, "toids":toids}, f)
    with open("Temp/word_dict.json", "w") as f:
        json.dump(all_dict, f)
    print("文件保存成功".center(30, "="))