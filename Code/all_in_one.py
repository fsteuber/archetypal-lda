import re
from collections import defaultdict
import numpy as np
from py_pcha.PCHA import PCHA
import scipy.sparse
from multiprocessing import Queue, Process
import argparse
import bz2
import os
from os import listdir
from os.path import isfile, join
import guidedlda
import itertools
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import entropy
import random
from sklearn.decomposition import PCA
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split

cores = 1
n_topics = (100, 200, 300)
# n_topics = (30, )  # debug
n_top_words = 20
s_conf = 0.7
path = "./"
n_iter = 500
allow_non_latin = False
n_files = 1
file_names = []
tweets_per_file = 50000
seed_confs = (0.5, 0.7, 0.9, 1)

def create_bow():
    global cores
    global path
    global n_files
    global file_names

    def run_insert(qw, qf, qm, n_workers):

        def run(qw, qm):
            bow_worker = defaultdict(int)  # dictionary holding token occurrences in tweets
            bow_hashtags = defaultdict(int)
            has_work = True
            while has_work:
                tweet = qw.get()
                if tweet is None:
                    has_work = False
                    continue
                tweet = tweet.strip()
                if tweet[0] == "[":  # hashtag input
                    tweet = tweet[1:-1]  # remove brackets
                    if not allow_non_latin:
                        tweet = "".join(re.findall("[a-zA-Z0-9 @#,:/._\-()]", tweet))
                    for ht in tweet.split(","):
                        ht = ht[1:-1]  # remove quotes
                        if len(ht) < 2:  # ht got filtered because of non latin chars
                            continue
                        bow_hashtags[ht] += 1
                else:  # tweet input
                    tweet = tweet[1:-1]  # remove quotes
                    if not allow_non_latin:
                        tweet = "".join(re.findall("[a-zA-Z0-9 @#,:/._\-()]", tweet))
                    for token in [w for w in tweet.split(" ") if len(w) > 3]:
                        bow_worker[token] += 1
            qm.put(("tokens", bow_worker))
            qm.put(("hashtags", bow_hashtags))
            qm.put((None, None))

        workers = [Process(target=run, args=(qw, qm)) for _ in range(n_workers)]
        for w in workers:
            w.start()

        has_work = True
        while has_work:
            file = qf.get()
            if file is None:
                has_work = False
                continue

            text_file = bz2.open(path + "data/" + file + ".text", "rt").readlines()
            hashtag_file = bz2.open(path + "data/" + file + ".hashtags", "rt").readlines()

            for i, (t, h) in enumerate(zip(text_file, hashtag_file)):
                if i % 25000 == 0:
                    print("processed " + str((100.0 * i) / len(text_file)) + "% of " + file, end="\r")
                qw.put(t)
                qw.put(h)
        for _ in range(n_workers):
            qw.put(None)

    tweet_files = [f for f in listdir(path + "data/") if isfile(join(path + "data/", f)) and f[-4:] == ".bz2"]
    random.shuffle(tweet_files)  # random selection of tweets every time

    q_worker = Queue(50000)
    q_master = Queue(cores*3)  # master queue handles words
    q_files = Queue(len(tweet_files)*2)

    for file in tweet_files[:min(n_files, len(tweet_files))]:
        file_names.append(file)
        q_files.put(file)

    n_fw = min(max(1, int(cores/3)), len(file_names))
    n_w = int((cores - n_fw)/n_fw)
    print("starting " + str(n_fw) + " file worker(s) and " + str(n_w*n_fw) + " tweet worker(s)")
    file_workers = [Process(target=run_insert, args=(q_worker, q_files, q_master, n_w)) for _ in range(n_fw)]
    for w in file_workers:
        w.start()
    for _ in file_workers:
        q_files.put(None)

    remaining_nones = n_w * n_fw
    bow_tokens = defaultdict(int)  # dictionary holding token occurrences in tweets
    bow_hashtags = defaultdict(int)  # dictionary holding hashtag occurrences in tweets

    while remaining_nones > 0 :
        print("waiting for " + str(remaining_nones) + " more workers to join              ", end="\r")
        t, worker_dict = q_master.get()

        if worker_dict is None:
            remaining_nones -= 1
            continue

        if t == "tokens":
            for k,v in worker_dict.items():
                bow_tokens[k] += v
        else:
            for k,v in worker_dict.items():
                bow_hashtags[k] += v

    print("writing results to file..            ")

    s = sorted([(v, k) for k, v in bow_tokens.items() if v > 10], reverse=True)  # sort tokens by occurrence count
    s = s[int(len(s) * 0.1):int(len(s) * 0.7)]  # remove language specific stop words and noise
    s = [k for v, k in s]  # retain tokens

    h = sorted([(v, k) for k, v in bow_hashtags.items() if v > 5], reverse=True)
    h = h[int(len(h) * 0.1):int(len(h) * 0.7)]
    h = [k for v, k in h]
    n_hts = len(h)

    f = open(path + '100-hashtags.txt', 'w')
    for k in h:
        f.write(k + '\n')
    f.close()

    print("BoW Size: " + str(len(s)) + " Word Token, " + str(n_hts) + " Hashtags")

    h.extend(s)  # the final bow vector consists of both, hashtags and conventional tokens

    f = open(path + '100-bow.txt', 'w')
    for k in h:
        f.write(k + '\n')
    f.close()

    return h, n_hts


def term_document_matrix(bow=None):
    global cores
    global path
    global file_names
    global tweets_per_file

    def run_insert(qw, qf, qm, bow, n_workers):
        global file_names
        global tweets_per_file

        def run(qw, qm, bow, tweets):
            # lil matrices are more efficient with row + col slicing than a csr/csc matrix would be
            worker_mat = scipy.sparse.lil_matrix((tweets, len(bow)), dtype=np.int8)
            has_work = True
            while has_work:
                i, tweet = qw.get()

                if tweet is None:
                    has_work = False
                    continue

                tweet = tweet.strip()
                if tweet[0] == "[":  # hashtag input
                    tweet = tweet[1:-1]  # remove brackets
                    if not allow_non_latin:
                        tweet = "".join(re.findall("[a-zA-Z0-9 @#,:/._\-()]", tweet))
                    for ht in tweet.split(","):
                        ht = ht[1:-1]  # remove quotes
                        if len(ht) < 2:  # ht got filtered because of non latin chars
                            continue
                        try:
                            j = bow.index(ht)
                            worker_mat[i, j] += 1
                        except:  # word not in bow
                            continue
                else:  # tweet input
                    tweet = tweet[1:-1]  # remove quotes
                    if not allow_non_latin:
                        tweet = "".join(re.findall("[a-zA-Z0-9 @#,:/._\-()]", tweet))
                    for token in [w for w in tweet.split(" ") if len(w) > 3]:
                        try:
                            j = bow.index(token)
                            worker_mat[i, j] += 1
                        except:  # word not in bow
                            continue
            qm.put(worker_mat.tocsr())
            qm.put(None)

        workers = [Process(target=run, args=(qw, qm, bow, len(file_names) * tweets_per_file)) for _ in range(n_workers)]
        for w in workers:
            w.start()

        has_work = True
        while has_work:
            file = qf.get()
            if file is None:
                has_work = False
                continue

            offset = file_names.index(file) * tweets_per_file
            text_file = bz2.open(path + "data/" + file + ".text", "rt").readlines()
            hashtag_file = bz2.open(path + "data/" + file + ".hashtags", "rt").readlines()

            for i, (t, h) in enumerate(zip(text_file, hashtag_file)):
                if i % 10000 == 0:
                    print("processed " + str((100.0 * i) / len(text_file)) + "% of " + file, end="\r")
                qw.put((offset + i, t))
                qw.put((offset + i, h))
        for _ in range(n_w):
            qw.put((None, None))

    if bow is None:
        bow = [x.strip() for x in open(path + '100-bow.txt', 'r').readlines()]

    q_worker = Queue(30000)
    q_master = Queue(cores*3)
    q_files = Queue(len(file_names) * 2)

    for file in file_names:
        q_files.put(file)

    n_fw = min(max(1, int(cores / 10)), len(file_names))
    n_w = int((cores - n_fw)/n_fw)

    print("starting " + str(n_fw) + " file worker(s) and " + str(n_w*n_fw) + " tweet worker(s)")
    file_workers = [Process(target=run_insert, args=(q_worker, q_files, q_master, bow, n_w)) for _ in range(n_fw)]
    for w in file_workers:
        w.start()

    for _ in file_workers:
        q_files.put(None)

    remaining_nones = n_w * n_fw
    X = scipy.sparse.csr_matrix((len(file_names) * tweets_per_file, len(bow)), dtype=np.int8)
    print("shape of TDM: " + str(X.shape))

    while remaining_nones > 0 :
        print("waiting for " + str(remaining_nones) + " more workers to join                  ", end="\r")
        mat = q_master.get()
        if mat is None:
            remaining_nones -= 1
            continue
        X += mat  # addition on lil matrices takes forever, so we're performing the merge operation on csr matrices

    scipy.sparse.save_npz(path + '200-feat_samples.npz', X)
    return X


def reduce_dim(tdm):
    global path

    # TDM preprocessing
    tdm = tdm[:, :n_hts]  # tdm for hashtags only
    tdm = tdm[tdm.getnnz(1) > 0]  # remove all zero rows from hashtag-tdm (no explanatory power)
    tdm = tdm.toarray()  # converting to dense matrix for PCA/PCHA
    print('shape of preprocessed Hashtag-Document Matrix: ' + str(tdm.shape))

    # find a good approximation for the number of principal components to use
    """
    exploratory_pca = PCA()
    exploratory_pca.fit(tdm)
    plt.plot(np.cumsum(exploratory_pca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.title('pca_var_hashtags')
    plt.savefig(path + 'pca_var_hashtags.pdf')
    npc = np.where(np.cumsum(exploratory_pca.explained_variance_ratio_) > 0.8)[0][0]
    """

    # Dimensionality Reduction
    pca = PCA(n_components=2000)
    pc = pca.fit_transform(tdm)
    np.save(path + '200-feat_samples_principal_components.npy', pc)
    pickle.dump(pca, open(path + 'model_pca.pickle', 'wb'))  # save model for later

    return pca, pc


def convex_hull_approximation(pca_model, n, bow, pc):  # n_arch = n_topics
    global path

    ####
    #pc = np.load(path + '200-feat_samples_principal_components.npy')
    #pca_model = pickle.load(open(path + "model_pca.pickle", "rb"))
    #bow = [x.strip() for x in open(path + "100-bow.txt").readlines()]
    #print(bow[:10])
    #XB = np.load(path + "topics_" + str(n) + "/seeded/XB.npy")

    ####
    train, test = train_test_split(pc, train_size=.4, shuffle=True)  # no class labels for stratification
    print(train.shape)

    XB, _, _, _, varexpl = PCHA(np.transpose(train), n, maxiter=100)
    np.save(path + 'topics_' + str(n) + "/seeded/XB.npy", XB)  # XB: Archetypes x Principal Components
    print("explained variance with " + str(n) + "archetypes:" + str(varexpl))


    Y = pca_model.inverse_transform(np.transpose(XB))  # transform back to original feature space


    f = open(path + 'topics_' + str(n) + '/seeded/300-archetypal_hashtags.csv', 'w')
    for i in range(n):
        f.write(','.join([bow[x] for x in np.where(np.argmax(Y, axis=0)==i)[0]]) + '\n')  # max pooling
    f.close()


def lda_non_seeded(n, bow, X):
    global n_iter
    global n_top_words
    model = guidedlda.GuidedLDA(n_topics=n, n_iter=n_iter)
    model.fit(X)

    print("saving model's properties")
    # Word Probabilities per Topic
    tdists = model.components_
    with bz2.open(path + 'topics_' + str(n) + '/non_seeded/400-topic_word_probabilities.csv.bz2', 'wt') as f:
        for tdist in tdists:
            f.write(','.join([str(x) for x in tdist]) + '\n')
        f.close()
    del f

    # Top Words per Topic
    f = open(path + 'topics_' + str(n) + '/non_seeded/400-topic_top_words.csv', 'w')
    for dist in tdists:
        bow = np.asarray(bow)
        elts = bow[np.argsort(dist)][:-31:-1]
        f.write(','.join(elts) + '\n')
    f.close()
    del f

    # Topic Distributions per Tweet
    with bz2.open(path + 'topics_' + str(n) + '/non_seeded/400-tweet_topic_distributions.csv.bz2', 'wt') as f:
        for row in model.doc_topic_:
            f.write(','.join([str(x) for x in row]) + '\n')
        f.close()
    del f

    print("calculating model's coherence and perplexity")
    # Coherence with n_top_words top words
    # tweet_topics = np.argmax(model.doc_topic_, axis=1)
    X = X.toarray()  # convert to dense matrix
    doc_freq = np.count_nonzero(X, axis=0)
    f = open(path + 'topics_' + str(n) + '/non_seeded/500-coherence_per_topic.txt', 'w')

    for dist in tdists:  # for each topic
        top_w = np.argsort(dist)[:-(n_top_words + 1):-1]  # select top words
        coherence = 0
        for i in range(1, len(top_w)):
            for j in range(0, i):
                coherence += np.log((np.count_nonzero(X[:, top_w[i]] * X[:, top_w[j]]) + 1 )/doc_freq[top_w[j]])
        f.write(str(coherence) + "\n")
    f.close()
    del f

    # Perplexity
    f = open(path + 'topics_' + str(n) + '/non_seeded/500-perplexity_per_topic.txt', 'w')
    for tdist in tdists:
        tdist = np.sort(tdist)[::-1]
        f.write(str(2 ** entropy(tdist)) + '\n')
    f.close()
    del f

    # number of words per topic
    f = open(path + 'topics_' + str(n) + '/non_seeded/500-n_words_per_topic.txt', 'w')
    for tdist in tdists:
        tdist = np.sort(tdist)[::-1]
        f.write(str(np.where(tdist > tdist[-1])[0][-1] + 1) + '\n')
    f.close()

    archetypal_takeover_rate(n, s)


def lda_seeded(n, bow, X, s):
    global s_conf
    global n_iter
    global n_top_words

    seeds = [x.strip().split(',') for x in
             open(path + 'topics_' + str(n) + '/seeded/300-archetypal_hashtags.csv').readlines()]
    seed_topics = {}
    word2id = dict((v, i) for i, v in enumerate(bow))

    for t_id, st in enumerate(seeds):
        for word in st:
            if len(word) <= 1:
                continue
            seed_topics[word2id[word]] = t_id

    model = guidedlda.GuidedLDA(n_topics=n, n_iter=(n_iter + 200))
    model.fit(X, seed_topics=seed_topics, seed_confidence=s)

    print("saving model's properties")
    # Word Probabilities per Topic
    tdists = model.components_
    with bz2.open(path + 'topics_' + str(n) + '/seeded/400-topic_word_probabilities_' + str(s) + '.csv.bz2', 'wt') as f:
        for tdist in tdists:
            f.write(','.join([str(x) for x in tdist]) + '\n')
        f.close()
    del f

    # Top Words per Topic
    f = open(path + 'topics_' + str(n) + '/seeded/400-topic_top_words_' + str(s) + '.csv', 'w')
    for dist in tdists:
        bow = np.asarray(bow)
        elts = bow[np.argsort(dist)][:-31:-1]
        f.write(','.join(elts) + '\n')
    f.close()
    del f

    # Topic Distributions per Tweet
    with bz2.open(path + 'topics_' + str(n) + '/seeded/400-tweet_topic_distributions_' + str(s) + '.csv.bz2', 'wt') as f:
        for row in model.doc_topic_:
            f.write(','.join([str(x) for x in row]) + '\n')
        f.close()
    del f

    print("calculating model's basic goodness features")
    # Coherence with n_top_words top words
    # tweet_topics = np.argmax(model.doc_topic_, axis=1)
    X = X.toarray()  # convert to dense matrix
    doc_freq = np.count_nonzero(X, axis=0)
    f = open(path + 'topics_' + str(n) + '/seeded/500-coherence_per_topic_' + str(s) + '.txt', 'w')

    for dist in tdists:  # for each topic
        top_w = np.argsort(dist)[:-(n_top_words + 1):-1]  # select top words
        coherence = 0
        for i in range(1, len(top_w)):
            for j in range(0, i):
                coherence += np.log((np.count_nonzero(X[:, top_w[i]] * X[:, top_w[j]]) + 1 )/doc_freq[top_w[j]])
        f.write(str(coherence) + "\n")
    f.close()
    del f

    # Perplexity
    f = open(path + 'topics_' + str(n) + '/seeded/500-perplexity_per_topic_' + str(s) + '.txt', 'w')
    for tdist in tdists:
        tdist = np.sort(tdist)[::-1]
        f.write(str(2 ** entropy(tdist)) + '\n')
    f.close()
    del f

    # number of words per topic
    f = open(path + 'topics_' + str(n) + '/seeded/500-n_words_per_topic_' + str(s) + '.txt', 'w')
    for tdist in tdists:
        tdist = np.sort(tdist)[::-1]
        f.write(str(np.where(tdist > tdist[-1])[0][-1] + 1) + '\n')
    f.close()

    archetypal_takeover_rate(n, s)


def archetypal_takeover_rate(n_topics, sconf):
    hashtag_bow = [x.strip() for x in open(path + '100-hashtags.txt').readlines()]
    hashtag_assignments = [x.strip().split(',') for x in
                           open(path + 'topics_' + str(n_topics) + '/seeded/300-archetypal_hashtags.csv').readlines()]
    if sconf == 0:  # non seeded LDA
        out = path + 'topics_' + str(n_topics) + '/non_seeded/500-hashtag_takeover.csv'
        X = np.loadtxt(path + 'topics_' + str(n_topics) +
                                          '/non_seeded/400-topic_word_probabilities.csv.bz2', delimiter=",", dtype=float)
    else:
        out = path + 'topics_' + str(n_topics) + '/seeded/500-hashtag_takeover_' + str(sconf) + '.csv'
        X = np.loadtxt(path + 'topics_' + str(n_topics) +
                       '/seeded/400-topic_word_probabilities_' + str(sconf) + '.csv.bz2', delimiter=",", dtype=float)

    X = np.transpose(X)[:len(hashtag_bow)]  # yields a n_hashtags x n_topics matrix
    X = np.argmax(X, axis=1)  # yields a n_hashtags x 1 Matrix with col 1 being the number of the archetype
    correct = []

    for i, arch_hts in enumerate(hashtag_assignments):
        c = 0
        err = False
        for ht in arch_hts:
            try:
                if X[hashtag_bow.index(ht)] == i:  # hashtag words to hashtag ids
                    c += 1
            except:
                err = True
        if not err:
            correct.append(c * 1.0 / len(arch_hts))

    with open(out, "w") as f:
        for c in correct:
            f.write(str(c) + "\n")


if __name__ == '__main__':
    sns.set()
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", dest='n_cores', type=int, default=[4], nargs=1,
                        help="number of cores to use for pre-processing and evaluation (anything between 1 and 32). "
                             "Default: 4")

    parser.add_argument("-p", dest='path', type=str, default=['./'], nargs=1,
                        help="path to working folder. Default: ./")

    parser.add_argument("-a", dest='all', action="store_true",
                        help="process any character from tweets (instead of latin chars only). Default: False")

    parser.add_argument("-f", dest='n_files', type=int, default=[1], nargs=1,
                        help="maximum number of files to work with. Default: 1")
    parser.add_argument("-m", dest='message', type=str, default=[''], nargs=1,
                        help="add a comment to the current execution")

    args = parser.parse_args()

    cores = args.n_cores[0]
    cores = 1 if cores < 1 else cores
    cores = 48 if cores > 48 else cores

    n_files = args.n_files[0]
    n_files = 1 if n_files < 1 else n_files
    n_files = 10**5 if n_files > 10**5 else n_files

    allow_non_latin = args.all

    path = args.path[0]

    for n in n_topics:  # automatically create result folders for each topic
        tmp = path + "topics_" + str(n) + "/"
        if not os.path.exists(tmp):
            os.makedirs(tmp)
        if not os.path.exists(tmp + "non_seeded/"):
            os.makedirs(tmp + "non_seeded/")
        if not os.path.exists(tmp + "seeded/"):
            os.makedirs(tmp + "seeded/")

    tweets = None
    bow = None
    n_hts = None
    tdm = None
    pca_model = None
    pc = None

    print('\nStep 1: Creating BoW File (DOP=' + str(cores) + ")")
    bow, n_hts = create_bow()

    print('Step 2: Creating Term-Document Matrix (DOP=' + str(cores) + ")")
    tdm = term_document_matrix(bow)

    bow = [x.strip() for x in open(path + '100-bow.txt').readlines()]
    tdm = scipy.sparse.load_npz(path + '200-feat_samples.npz')

    print('Step 3: Calculating Archetypal Hulls')
    pca_model, pc = reduce_dim(tdm)

    pcha_workers = [Process(target=convex_hull_approximation, args=(pca_model, n, bow, pc)) for n in n_topics]
    for w in pcha_workers:
        w.start()
    for w in pcha_workers:
        w.join()
    del pcha_workers

    print('Steps 4 + 5: Train GuidedLDA and calculate Perplexity and Coherence (DOP=' + str(len(n_topics) * 2) + ")")
    lda_workers = []

    for n in n_topics:
        for s in seed_confs:
            lda_workers.append(Process(target=lda_seeded, args=(n, bow, tdm, s)))

    lda_workers = [Process(target=lda_seeded, args=(n, bow, tdm)) for n in n_topics]
    lda_workers.extend([Process(target=lda_non_seeded, args=(n, bow, tdm)) for n in n_topics])

    for w in lda_workers:
        w.start()
    for w in lda_workers:
        w.join()
    del lda_workers

    print('Step 6: Evaluating Hashtag Takeover Rate')
    workers = []
    for n in n_topics:
        for s in (0, 0.5, 0.7, 0.9, 1):
            workers.append(Process(target=archetypal_takeover_rate, args=(n, s)))
    for w in workers:
        w.start()
    for w in workers:
        w.join()
