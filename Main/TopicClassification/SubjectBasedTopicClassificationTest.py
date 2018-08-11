#-*- coding: utf-8 -*-
from Main.Model.models import CNN
from Main.Model.data_process import sentence_to_index_morphs
import tensorflow as tf
import numpy as np
import re, json, os, datetime

class TopicClassifier:
    def __init__(self):
        DIR = '../../data/Subject_models'
        keyword = '망나니'

        # load vocab, vocab_size, max_length
        with open('vocab.json', 'r') as fp:
            vocab = json.load(fp)

        # load configuration
        with open('config.txt', 'r') as f:
            vocab_size = int(re.sub('\n', '', f.readline()))
            max_length = int(f.readline())

        # open session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        # make model instance
        model = CNN(sess=sess, vocab_size=vocab_size, sequence_length=max_length, trainable=True)

        # load trained model
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(os.path.join(DIR)))

        # inference
        topic = sentence_to_index_morphs(keyword, vocab, max_length)

        prediction, prob = model.predict(topic)

        for i in range(len(prediction)):
            timestamp = datetime.datetime.now()
            if prob[i] < 0.2:
                timestamp = datetime.datetime.now()
                print('TIMESTAMP', '\t%s' % timestamp, '\nAccuracy:', '%05f' % (prediction[i]), '\tloss:',
                      '%05f' % (1 - prob[i]))
            else:
                print('TIMESTAMP', '\t%s' % timestamp, '\nAccuracy:', '%05f' % (prediction[i]), '\tloss:',
                      '%05f' % (1 - prob[i]))

            # save configuration
            with open('../../data/TopicClassification/[Classification]Result.txt', 'a') as f:
                f.write('TIMESTAMP \t%s \nAccuracy: %05f \tloss: %05f \n' % (timestamp, prediction[i], 1 - prob[i]))

if __name__ == "__main__":
    TopicClassifier()
