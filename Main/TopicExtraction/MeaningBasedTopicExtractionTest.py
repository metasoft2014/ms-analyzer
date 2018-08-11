#-*- coding: utf-8 -*-
from Main.Model.models import logistic_regression
from Main.Model.data_process import sentence_to_index_morphs
import tensorflow as tf
import pandas as pd
import numpy as np
import re, json, os, datetime

class TopicExtractor:
    def __init__(self):
        self.DIR = '../../data/Meaning_models'

    def extract(self):
        # load vocab, vocab_size, max_length
        with open('tagged_data.json', 'r') as fp:
            tagged_data = json.load(fp)

        index = []

        for item in tagged_data:
            index.append(item[1])

        # open session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        # make model instance
        num_epochs_per_cycle = 5
        num_cycles = 10
        vec_size = 1
        alpha = 1e-1
        min_alpha = 1e-3
        model = logistic_regression(sess=sess, vocab_size=vec_size, lr=1e-2)

        # load trained model
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(os.path.join(self.DIR)))

        prediction, prob = model.predict(index)

        timestamp = datetime.datetime.now()
        print('TIMESTAMP', '\t%s' % timestamp, '\nAccuracy:', '%05f' % np.mean(prob), '\tloss:',
              '%05f' % (1 - np.mean(prob)))

        # save configuration
        with open('../../data/TopicExtraction/[Extraction]Result.txt', 'a') as f:
            f.write('TIMESTAMP \t%s \nAccuracy: %05f \tloss: %05f \n' % (timestamp, np.mean(prob), 1 - np.mean(prob)))

if __name__ == "__main__":
    extractor = TopicExtractor()
    extractor.extract()
