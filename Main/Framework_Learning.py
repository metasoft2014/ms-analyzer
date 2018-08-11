#-*- coding: utf-8 -*-
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

import numpy as np
from Main.Model.data_process import build_vocab_pos, sentence_to_onehot_morphs, cal_idf_morphs, sentence_to_tfidf_morphs
import os
import csv

from Main.Resources.Database import mariadb

if __name__ == "__main__":
    CONFIGDIR = './'
    DIR = '../data/Dictionary'

    # build dataset
    db = mariadb(CONFIGDIR)
    df = db.get_content_from_db('ms_collected_data')

    data = df.content

    vocab, _, vocab_size = build_vocab_pos(data)

    w = csv.writer(open(os.path.join(DIR, "[Dic]RelatedWords.csv"), "w"))
    w.writerow(['Total Count', len(vocab)])

    for key, val in vocab.items():
        w.writerow([key, val])

    sentence_to_onehot_morphs(data, vocab)

    IDF = cal_idf_morphs(data, vocab)

    tfidf = sentence_to_tfidf_morphs(data, vocab, IDF)

    np.savetxt(os.path.join(DIR,'[Dic]RelatedWords_IDF.csv'), tfidf, delimiter=',')