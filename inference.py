from gensim.models import Word2Vec
import json
import tensorflow as tf
import jieba

if __name__ == "__main__":
    corpus_size = 62
    # 读取字典和词向量矩阵
    with open("Temp/word_dict.json", "w") as f:
        all_dict = json.load(f)
    words2id = {j: i for i, j in enumerate(all_dict)}
    id2words = {i: j for i, j in enumerate(all_dict)}

    model = Word2Vec.load("Temp/word2vec.model")
    emb_size = model.layer1_size
    max_inference_sequence_length = 30
    loaded_graph = tf.Graph()
    with tf.Session(loaded_graph) as sess:
        # 导入计算图参数
        loader = tf.train.import_meta_graph(meta_graph_or_file="./chechpoints/training_mode.ckpt.meta")
        loader.restore(sess, "./chechpoints/training_model.ckpt")
        input_data = loaded_graph.get_tensor_by_name("source:0")
        logits = loaded_graph.get_tensor_by_name("inference_logits:0")
        source_sequence_length = loaded_graph.get_tensor_by_name("source_sequence_length:0")
        emb_matrix = loaded_graph.get_tensor_by_name("embedding_matrix:0")
        input_words = "你好"
        input_ids = [words2id.get(i, words2id["_UNK"]) for i in jieba.lcut(input_words)]
        answer_logits = sess.run(logits, feed_dict={input_data: [input_ids]*corpus_size,
                                                    source_sequence_length:[len(input_ids)]*corpus_size})