import os
import jieba

if __name__ == '__main__':
    # 读取数据
    corpus_path = "/Data"
    corpus_files = os.listdir(corpus_path)
    corpus = []
    for corpus_file in corpus_files:
        with open(os.path.join(corpus_path, corpus_file), encoding="utf8") as f:
            corpus.extend(f.readlines())
    corpus = [str.strip(i) for i in corpus]  # 去除空格和"\n"
    print("语料库读取完成".center(30, "="))
    # 分词并构建词典
    jieba.load_userdict("")
    corpus_cut = [jieba.lcut(i) for i in corpus]
    print("分词完成".center(30, "="))
    # 构建词典单词与数值id的映射关系

