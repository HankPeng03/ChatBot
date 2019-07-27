import json
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, MultiRNNCell
def get_lstm_cell(hidden_size):
    lstm_cell =  LSTMCell(num_units=hidden_size,
                          initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1, seed=123))
    return lstm_cell

def my_encoder(hidden_size, num_layers, embedding_matrix, input_data, input_data_length):
    encoder_embedding_input = tf.nn.embedding_lookup(params=embedding_matrix, ids=input_data)
    encoder_cells = MultiRNNCell([get_lstm_cell(hidden_size) for i in range(num_layers)])
    encoder_output, encoder_state = tf.nn.dynamic_rnn(cell=encoder_cells,
                      inputs=encoder_embedding_input,
                      sequence_length=input_data_length,
                      dtype=tf.float32)
    return encoder_output, encoder_state

def my_decoder(output_data, corpus_size, word2id, embedding_matrix, hidden_size,num_layers,
               vocab_size, output_sequence_length, max_output_sequence_length, encoder_output):
    # Decoder
    ## 添加_BOS
    ending = tf.strided_slice(output_data, begin=[0,0], end=[corpus_size, -1], strides=[1,1])
    begin_signal = tf.fill(dims=[corpus_size, 1], value=word2id["_BOS"])
    decoder_input_data = tf.concat([begin_signal, ending], axis=1, name="decoder_input_data")

    decoder_embedding_input = tf.nn.embedding_lookup(params=embedding_matrix, ids=decoder_input_data)
    decoder_cells = MultiRNNCell([get_lstm_cell(hidden_size) for i in range(num_layers)])

    # Projection Layer
    projection_layer = tf.layers.Dense(units=vocab_size,
                    kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

    # Training Decoder
    with tf.variable_scope("Decoder"):
        # Helper对象
        training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_embedding_input,
                                                            sequence_length=output_sequence_length)
        # Basic Decoder
        training_decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cells,
                                        helper=training_helper,
                                        initial_state=decoder_cells.zero_state(batch_size=corpus_size, dtype=tf.float32),
                                        output_layer=projection_layer)
        # Dynamic Decoder
        training_final_output, training_final_state, training_sequence_length = tf.contrib.seq2seq.dynamic_decode(decoder=training_decoder,
                                          maximum_iterations=max_output_sequence_length,
                                          impute_finished=True)

    # Inference Decoder
    with tf.variable_scope("Decoder", reuse=True):
        # Helper对象
        start_tokens = tf.tile(input=tf.constant(value=[word2id["_BOS"]], dtype=tf.int32),
                               multiples=[corpus_size], name="start_tokens")
        inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding=emb_matrix,
                                                 start_tokens=start_tokens,
                                                 end_token=word2id["_EOS"])
        # Basic Decoder
        inference_decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cells,
                                                            helper=inference_helper,
                                                            initial_state=decoder_cells.zero_state(corpus_size, dtype=tf.float32),
                                                            output_layer=projection_layer)
        # Dynamic Decoder
        inference_final_output, inference_final_state, inference_sequence_length = tf.contrib.seq2seq.dynamic_decode(decoder=inference_decoder, imput_finished=True,
                                          naximun_iterations=max_output_sequence_length)
    return training_final_output, training_final_state, inference_final_output, inference_final_state

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
    max_input_sequence_length = max(from_length)
    source = [i + [word2id["_PAD"]]*(max_input_sequence_length-len(i)) for i in fromids]

    to_length = [len(i) for i in toids]
    max_output_sequence_length = max(to_length)
    target = [i + [word2id['_PAD']]*(max_output_sequence_length-len(i)) for i in toids]
    # 定义Tensor

    num_layers = 2
    hidden_size = 100
    learning_rate = 0.001
    batch_size = 64

    ## 设置占位符
    input_data = tf.placeholder(dtype=tf.int32, shape=[corpus_size, None], name="source")  # 输入
    output_data = tf.placeholder(dtype=tf.int32, shape=[corpus_size, None], name="target")  # 输出
    input_sequence_length = tf.placeholder(dtype=tf.int32, shape=[corpus_size], name="source_sequence_length")  # 输入句子的长度
    output_sequence_length = tf.placeholder(dtype=tf.int32, shape=[corpus_size], name="target_sequence_length")  # 输出句子的长度
    emb_matrix = tf.constant(embedding_matrix, name="embedding_matrix", dtype=tf.float32)  # 词向量矩阵

    # Encoder
    encoder_output, encoder_state = my_encoder(hidden_size, num_layers, embedding_matrix, input_data, input_sequence_length)

    # Decoder
    training_final_output, training_final_state, inference_final_output, inference_final_state = \
        my_decoder(output_data, corpus_size, word2id, embedding_matrix, hidden_size, num_layers,
               vocab_size, output_sequence_length, max_output_sequence_length, encoder_output)

    # Encoder-Decoder Model（Seq2Seq Model）

    ##  Loss Function
    training_logits = tf.identity(input=training_final_output.rnn_output, name="training_logits")
    inference_logits = tf.identity(input=inference_final_output.sample_id, name="inference_logits")
    mask = tf.sequence_mask(lengths=output_sequence_length, maxlen=max_output_sequence_length,
                            name="mask", dtype=tf.float32)
    cost = tf.contrib.seq2seq.sequence_loss(logits=training_logits, targets=output_data,
                                     weights=mask)
    ## Optimizer （自适应优化器）
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    ## 梯度剪枝（防止梯度爆炸或梯度消失）
    gradients = optimizer.compute_gradients(cost)
    clip_gradients = [(tf.clip_by_value(t=grad, clip_value_max=5.0, clip_value_min=-5.0), var)
                      for grad, var in gradients if grad is not None]
    train_op = optimizer.apply_gradients(clip_gradients)

    # Train
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        _, training_pred = sess.run(train_op, feed_dict={
            input_data: source,
            output_data: target,
            input_sequence_length: from_length,
            output_sequence_length: to_length
        })

        # print(' '.join([id2word[i] for i in source[0] if i != word2id["_EOS"]]))
        # print(' '.join([id2word[i] for i in target[0] if i != word2id["_EOS"]]))
        # print(' '.join([id2word[i] for i in training_pred[0] if i != word2id["_EOS"]]))
