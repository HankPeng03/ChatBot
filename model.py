import json
import tensorflow as tf

if __name__ == '__main__':
    # 文件读取
    with open("Temp/data.json", "r") as f:
        data = json.load(f)
    fromids = data["fromids"]
    toids = data["toids"]
    del data

    with open("Temp/word_dict.json", "r") as f:
        all_dict = json.load(f)

    word2id = {w: id for id, w in enumerate(all_dict)}
    id2word = {id: w for id, w in enumerate(all_dict)}
    # 构建词向量矩阵
    from gensim.models import Word2Vec
    model = Word2Vec.load("Temp/word2vec.model")

    import numpy as np
    vocab_size = len(all_dict)
    corpus_size = len(fromids)
    emb_size = 100
    embedding_matrix = np.zeros((vocab_size, emb_size), dtype=np.float32)
    tmp = np.diag([1]*emb_size)  # 对于出现在词典中，但是没有出现在词向量中的词汇，使用One-hot代替
    k = 0
    for i in range(vocab_size):
        try:
            embedding_matrix[i] = model.wv[str(i)]
        except:
            embedding_matrix[i] = tmp[k]
            k += 1
    # 统计id向量的长度，并统一长度
    from_length = [len(i) for i in fromids]
    m = max(from_length)
    source = [i + [word2id["_PAD"]]*(m-len(i)) for i in fromids]

    to_length = [len(i) for i in toids]
    m = max(to_length)
    target = [i + [word2id['_PAD']]*(m-len(i)) for i in toids]

    # 定义Tensor

    num_layers = 2
    hidden_size = 100
    learning_rate = 0.001

    ## 设置占位符
    input_data = tf.placeholder(dtype=tf.int32, shape=[corpus_size, None], name="source")  # 输入
    output_data = tf.placeholder(dtype=tf.int32, shape=[corpus_size, None], name="target")  # 输出
    input_data_length = tf.placeholder(dtype=tf.int32, shape=[corpus_size], name="source_sequence_length")  # 输入句子的长度
    output_data_length = tf.placeholder(dtype=tf.int32, shape=[corpus_size], name="target_sequence_length")  # 输出句子的长度
    emb_matrix = tf.constant(embedding_matrix, name="embedding_matrix", dtype=tf.float32)  # 词向量矩阵


    # Encoder

    # Decoder

    # Encoder-Decoder Model（Seq2Seq Model）

    ##  Loss Function

    ## Optimizer （自适应优化器）

    ## 梯度剪枝（防止梯度爆炸或梯度消失）

