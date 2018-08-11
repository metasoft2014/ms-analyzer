#-*- coding: utf-8 -*-
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import numpy as np
import gensim
import tensorflow as tf
import os, json, sys, datetime
from Main.Model.models import CNN
from Main.Model.data_process import morphs_process, sentence_to_index_morphs, batch_iter
from Main.Model.word2vec import make_embedding_vectors

from Main.Resources.Database import mariadb

if __name__ == "__main__":
    CONFIGDIR = '../'
    DIR = '../../data/Subject_models'
    RESULT_DIR = '../../data/TopicClassification'
    TABLE_NAME = 'ms_collected_data'

    # build dataset
    db = mariadb(CONFIGDIR)
    df = db.get_content_from_db(TABLE_NAME)
    data = df.content

    tokens = morphs_process(data)
    wv_model = gensim.models.Word2Vec(min_count=1, window=5, size=300)
    wv_model.build_vocab(tokens)
    wv_model.train(tokens, total_examples=wv_model.corpus_count, epochs=wv_model.epochs)
    word_vectors = wv_model.wv

    wv_pairs = word_vectors.most_similar(df.keyword[0], topn=data.size)

    str_list = []
    vector_list = []

    for item in wv_pairs:
        str_list.append(item[0])
        vector_list.append(1) if item[1] >= 0.8 else 0

    x_input = str_list
    y_input = vector_list
    max_length = 30

    print('데이터로부터 정보를 얻는 중입니다.')
    embedding, vocab, vocab_size = make_embedding_vectors(list(x_input))
    print('완료되었습니다.')

    # save vocab, vocab_size, max_length
    with open('vocab.json', 'w') as fp:
        json.dump(vocab, fp)

    # save configuration
    with open('config.txt', 'w') as f:
        f.write(str(vocab_size) + '\n')
        f.write(str(max_length))

    # open session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # make model instance
    model = CNN(sess=sess, vocab_size=vocab_size, sequence_length=max_length, trainable=True)

    # assign pretrained embedding vectors
    model.embedding_assign(embedding)

    # make train batches
    batches = batch_iter(list(zip(x_input, y_input)), batch_size=64, num_epochs=5)

    # model saver
    saver = tf.train.Saver(max_to_keep=3, keep_checkpoint_every_n_hours=0.5)

    # train model
    print('모델 훈련을 시작합니다.')
    avgLoss = []

    for step, batch in enumerate(batches):
        x_train, y_train = zip(*batch)

        x_train = sentence_to_index_morphs(x_train, vocab, max_length)
        l, _, acc = model.train(x_train, y_train)
        avgLoss.append(l)
        if step % 10 == 0:
            timestamp = datetime.datetime.now()
            print('batch:', '%04d' % step, '\nTIMESTAMP', '\t%s' % timestamp, 'loss:',
                  '%05f' % np.mean(avgLoss), 'Accuracy:', '%05f' % (acc))

            # save configuration
            with open('out.txt', 'a') as f:
                f.write('batch: %04d \nTIMESTAMP \t%s \nloss: %05f \tAccuracy: %05f \n' % (step, timestamp, np.mean(avgLoss), acc))

            saver.save(sess, os.path.join(DIR, "model"), global_step=step)
