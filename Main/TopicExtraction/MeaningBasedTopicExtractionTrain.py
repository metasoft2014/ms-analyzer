#-*- coding: utf-8 -*-
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.model_selection import train_test_split
from Main.Model.models import logistic_regression
from Main.Model.data_process import morphs_process, batch_iter
import tensorflow as tf
import numpy as np
import os, json

from Main.Resources.Database import mariadb

if __name__ == "__main__":
    CONFIGDIR = '../'
    DIR = '../../data/Meaning_models'
    TABLE_NAME = 'ms_collected_data'

    # build dataset
    db = mariadb(CONFIGDIR)
    df = db.get_content_from_db(TABLE_NAME)

    train, test = train_test_split(df, test_size=0.2)

    X_train = train.content
    Y_train = train.index
    X_test = test.content
    Y_test = test.index

    tagged_data = [TaggedDocument(words=_d, tags=[str(i)]) for i, _d in enumerate(X_train)]

    # save vocab, vocab_size, max_length
    with open('tagged_data.json', 'w') as fp:
        json.dump(tagged_data, fp)

    num_epochs_per_cycle = 5
    num_cycles = 10
    vec_size = 1
    alpha = 1e-1
    min_alpha = 1e-3
    dv_model = Doc2Vec(vector_size=vec_size, alpha=alpha, min_count=2, dm=0)
    dv_model.build_vocab(tagged_data)
    for cycle in range(num_cycles):
        dv_model.train(tagged_data, total_examples=dv_model.corpus_count,
                       epochs=num_epochs_per_cycle, start_alpha=alpha, end_alpha=min_alpha)
        print('cycle:', '%02d' % (cycle + 1))

    X_train_vector = []
    X_test_vector = []
    for i in range(len(X_train)):
        X_train_vector.append(dv_model.docvecs[str(i)])
    for i in range(len(X_test)):
        X_test_vector.append(dv_model.infer_vector(X_test.values[i], alpha=alpha, min_alpha=min_alpha, steps=5))

    # make train data
    batches = batch_iter(list(zip(X_train_vector, Y_train)), batch_size=64, num_epochs=15)

    # open session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.reset_default_graph()
    sess = tf.Session(config=config)

    # make model instance
    model = logistic_regression(sess=sess, vocab_size=vec_size, lr=1e-2)


    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []

    # model saver
    saver = tf.train.Saver(max_to_keep=3, keep_checkpoint_every_n_hours=0.5)

    # train model
    for step, batch in enumerate(batches):
        x_train, y_train = zip(*batch)
        acc = model.get_accuracy(x_train, y_train)
        l, _ = model.train(x_train, y_train)
        train_loss.append(l)
        train_acc.append(acc)

        if step % 100 == 0:
            test_batches = batch_iter(list(zip(X_test_vector, Y_test)), batch_size=64, num_epochs=1)
            for test_batch in test_batches:
                x_test, y_test = zip(*test_batch)
                t_acc = model.get_accuracy(x_test, y_test)
                t_loss = model.get_loss(x_test, y_test)
                test_loss.append(t_loss)
                test_acc.append(t_acc)
            print('batch:', '%04d' % step, '\ntest accuracy:',
                  '%.3f' % (1-np.mean(test_acc)), '\ntrain loss:', '%.5f' % np.mean(train_loss),
                  '\ttest loss:', '%.5f' % np.mean(test_loss))
            print('train accuracy:', '%.3f' % (1-np.mean(train_acc)), '\ttest accuracy:',
                  '%.3f' % (1-np.mean(test_acc)), '\n')
            saver.save(sess, os.path.join(DIR, "model"), global_step=step)
            train_loss = []
            train_acc = []
            test_loss = []
            test_acc = []