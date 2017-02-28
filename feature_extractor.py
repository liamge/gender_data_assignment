# extract features from list of text instances based on configuration set of features

import nltk
import numpy
import re
import time
import csv
import math
import pickle
import gensim
from numpy.linalg import svd
from nltk import ngrams
from collections import *
from sklearn.preprocessing import normalize
from gensim import corpora
from gensim.models.lsimodel import LsiModel


source_text = []
stemmed_text = []
model = gensim.models.Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)

def timeit(func):
    # Timeit is a function that can be used as a decorator for another function so you can see its run time

    def timed(*args, **kwargs):
        t1 = time.time()
        result = func(*args, **kwargs)
        t2 = time.time()

        print('Method {} took {} seconds'.format(func.__name__, t2-t1))
        return result

    return timed

@timeit
def preprocess():
    # first stem and lowercase words, then remove rare
    # lowercase
    global source_text
    source_text = [text.lower() for text in source_text]

    # tokenize
    global tokenized_text
    tokenized_text = [nltk.word_tokenize(text) for text in source_text]

    # POS tag
    global tagged_text
    tagged_text = [[n[1] for n in nltk.pos_tag(essay)] for essay in tokenized_text]

    # stem
    porter = nltk.PorterStemmer()
    global stemmed_text
    stemmed_text = [[porter.stem(t) for t in tokens] for tokens in tokenized_text]

    # remove rare
    vocab = nltk.FreqDist(w for line in stemmed_text for w in line)
    rarewords_list = set(vocab.hapaxes())
    stemmed_text = [['<RARE>' if w in rarewords_list else w for w in line] for line in stemmed_text]
    # note that source_text will be lowercased, but only stemmed_text will have rare words removed

@timeit
def bag_of_function_words():
    bow = []
    for sw in nltk.corpus.stopwords.words('english'):
        counts = [sum(1 for _ in re.finditer(r'\b%s\b' % sw, line)) for line in source_text]
        bow.append(counts)
    return bow, nltk.corpus.stopwords.words('english')

# FILL IN OTHER FEATURE EXTRACTORS

# NGRAM FUNCTIONS
@timeit
def pos_transform(text):
    # Turns tokenized text into a list of lists containing POS tags
    pos = [[x[1] for x in nltk.pos_tag(essay)] for essay in text]

    return pos


@timeit
def compute_ngrams(n, text, pos=False):
    # General function that takes a tokenized corpus as input and outputs a list of lists
    # with each sublist containing the bigrams in it's equivalent essay
    ngs = []
    if pos:
        # List comprehension for turning a text into a list of it's POSs
        text = tagged_text
    if n == 1 and not pos:
        # List comprehension for handling unigram stripping of stopwords
        text = [[w for w in essay if w not in nltk.corpus.stopwords.words('english')] for essay in text]
    for essay in text:
        # Because ngrams returns a generator, list(ngrams) will return a list of it's elements
        ngs.append(list(ngrams(essay, n)))
    return ngs, text


@timeit
def ngram_counts(n, top_n, text, pos=False):
    # Counts instances of ngrams, and returns the top n ngrams (confusing notation)
    if pos:
        ngs, text = compute_ngrams(n, text, pos=True)
    else:
        ngs, _ = compute_ngrams(n, text)
    counts = defaultdict(int)
    for essay in ngs:
        for bg in essay:
            counts[bg] += 1

    sl = sorted(counts.items(), key=lambda x: x[1], reverse=True)  # Sorted list in descending order
    top_ngrams = [x[0] for x in sl][:top_n]

    bow = []
    for ng in top_ngrams:
        counts = [float(sum(x == ng for x in ngrams(line, n))) / (len(line)) for line in text]
        bow.append(counts)
    return bow

# COMPLEXITY FUNCTIONS
@timeit
def characters_per_word(text):
    feats = []
    for essay in text:
        counts = [sum([len(w) for w in essay])/len(essay)]
        feats.append(counts)
    feats = numpy.asarray(feats).T.tolist() # To keep the dimensions consistent with features in extract_features
    return feats

@timeit
def words_per_sentence(text):
    feats = []
    essays = [nltk.sent_tokenize(essay) for essay in text]
    for essay in essays:
        counts = 0
        for sent in essay:
            counts += len(sent)
        counts = counts/len(essay)
        feats.append([counts])
    feats = numpy.asarray(feats).T.tolist()
    return feats

@timeit
def unique_words_ratio(text):
    feats = []
    for essay in text:
        feats.append([len(set(essay))/len(essay)])
    feats = numpy.asarray(feats).T.tolist()
    return feats

@timeit
def words_per_sentence(text):
    feats = []
    for essay in text:
        essay = nltk.sent_tokenize(essay)
        feats.append([sum([len(sent) for sent in essay])/len(essay)])
    feats = numpy.asarray(feats).T.tolist()
    return feats

# TOPIC MODELS
@timeit
def lsi_transform(text, n_topics):
    dictionary = corpora.Dictionary(text)
    corpus = [dictionary.doc2bow(essay) for essay in text]

    lsi = LsiModel(corpus=corpus, num_topics=n_topics)
    return lsi, dictionary


@timeit
def topic_models(text, n_topics):
    # Stopwords are uninformative for topic models
    text = [[w for w in essay if w not in nltk.corpus.stopwords.words('english')] for essay in text]
    lsi, dictionary = lsi_transform(text, n_topics)

    topics = []
    for essay in text:
        e2i = dictionary.doc2bow(essay)
        tps = list([x[1] for x in lsi[e2i]])
        topics.append(tps)

    topics = numpy.asarray(topics).T.tolist()
    return topics

# WORD VECTORS
def transform_sent(sent):
    '''
    :param sent: list of tokenized words
    :return: matrix of vectors corresponding to words in sentence (ones if <PAD> token)
    '''
    vector = []
    for w in sent:
        if w == '<PAD>' or w not in model.vocab:
            vector.append(numpy.ones((300,)))
        else:
            vector.append(model[w])
    vector = numpy.array(vector)

    return vector

def pad_sentences(text):
    '''
    :param text: list of lists where sublists contain tokenized words
    :return: list of lists where sublists contain fixed length sequences of words and <PAD> tokens
    '''
    maxlen = max([len(sent) for sent in text])
    for sent in text:
        while len(sent) < maxlen:
            sent.append('<PAD>')

    return text

def transform_tfidf(sent, weighted):
    '''
    :param sent: List of tokenized words
    :param weighted: Dictionary of weighted word vectors
    :return: the matrix where each row is a tf-idf weighted word vector
    '''
    vector = []
    words = [w for w in sent if w in model.vocab]
    for w in words:
        vector.append(weighted[w])
    vector = numpy.array(vector)

    return vector

def average_sent(sent):
    '''
    :param sent: list of tokenized words
    :return: averaged word vectors for every word in sent if word is in model vocab
    '''
    vecs = transform_sent(sent)
    mean = numpy.mean(vecs, axis=0)
    return mean

def average_tfidf(sent, weights):
    '''
    :param sent: List of tokenized words
    :param weights: Dictionary containing tf-idf weighted word vectors
    :return: Averaged word vectors
    '''
    vecs = transform_tfidf(sent, weights)
    mean = numpy.mean(vecs, axis=0)
    return mean

def tf(term, doc):
    '''
    :param term: string of desired term
    :param doc: list of tokenized words
    :return: number of instances of term in doc regularized by the length
    '''
    return doc.count(term) / len(doc)

def idf(term, docs):
    '''
    :param term: string of desired term
    :param docs: list of lists where sublists are lists of tokenized words
    :return: inverse doc frequency of the term w/r/t a corpus
    '''
    n_docs_containing = sum(term in d for d in docs)
    return math.log(len(docs) / (1 + n_docs_containing))

def tf_idf(term, doc, docs):
    '''
    :param term: string of desired term
    :param doc: list of tokenized words
    :param docs: list of lists where sublists are lists of tokenized words
    :return: tf-idf score for a term given a document and a corpus
    '''
    return tf(term, doc) * idf(term, docs)

def tf_idf_generation(text):
    '''
    :param text: list of lists where sublists are lists of tokenized words
    :return: dictionary of text vocab where values are their respective tf-idf scores
    '''
    doc = []
    for sent in text:
        doc.append([w for w in sent if w in model.vocab])

    weighted = defaultdict(float)
    for sent in doc:
        for w in sent:
            if w not in weighted:
                weighted[w] = tf_idf(w, sent, text) * model[w]

    return weighted

@timeit
def average_word_vecs(text, tfidf=False):
    '''
    :param text: text to return averaged word vectors for
    :return: averaged word vectors for document in text
    '''
    features = []
    if tfidf:
        weights = tf_idf_generation(text)
    for doc in text:
        if tfidf:
            features.append(average_tfidf(doc, weights))
        else:
            features.append(average_sent(doc))

    features = numpy.asarray(features).T.tolist()
    return features

def pointwise_mult(sent):
    '''
    :param sent: list of tokenized words
    :return: pointwise mutiplication of the corresponding word vectors
    '''
    vectors = transform_sent(sent)
    vec = vectors[0]
    for i in range(1, len(vectors)):
        vec = numpy.multiply(vec, vectors[i])

    return vec

@timeit
def pointwise_wrapper(text):
    '''
    :param text: list of lists where each sublist is of tokenized words
    :return: feature format
    '''
    features = []
    for doc in text:
        features.append(pointwise_mult(doc))

    features = numpy.array(features).T.tolist()
    return features

def svd_decomp(sent):
    '''
    :param sent: a tokenized sentence (or document)
    :returns: square matrix A' where A' = A x At
    '''
    matrix = transform_sent(sent)
    U, s, V = svd(matrix)

    return s.tolist()

@timeit
def svd_wrapper(text):
    features = []
    for i in range(len(text)):
        features.append(svd_decomp(text[i]))

    features = numpy.array(features).T.tolist()
    return features

def load_encoded():
    '''
    :return: encoded sentence vectors based off of the autoencoder in autoencoder.ipynb
    '''
    features = pickle.load(open('encoded.pkl', 'rb')).T.tolist()
    return features


def log(fvec, hvec):
    with open('log.csv', 'a') as lfile:
        lwriter = csv.writer(lfile)
        lwriter.writerow(hvec)
        lwriter.writerows(fvec)

def extract_features(text, conf):
    all = False
    if len(conf)==0:
        all = True

    global source_text
    source_text = text			# we'll use global variables to pass the data around
    preprocess()

    features = []		# features will be list of lists, each component list will have the same length as the list of input text
    header = []


    # extract requested features: FILL IN HERE
    if 'bag_of_function_words' in conf or all:
        fvec, hvec = bag_of_function_words()
        features.extend(fvec)
        header.extend(hvec)
        log(fvec, hvec)
    if 'encoded' in conf or all:
        features.extend(load_encoded())
    if 'svd_word_vectors' in conf or all:
        features.extend(svd_wrapper(pad_sentences(tokenized_text)))
    if 'pointwise_word_vectors' in conf or all:
        features.extend(pointwise_wrapper(tokenized_text))
    if 'average_word_vectors' in conf or all:
        features.extend(average_word_vecs(tokenized_text))
    if 'tfidf_word_vectors' in conf or all:
        features.extend(average_word_vecs(tokenized_text, tfidf=True))
    if 'bag_of_trigrams' in conf or all:
        features.extend(ngram_counts(3, 500, stemmed_text))
    if 'bag_of_bigrams' in conf or all:
        features.extend(ngram_counts(2, 100, stemmed_text))
    if 'bag_of_unigrams' in conf or all:
        features.extend(ngram_counts(1, 100, stemmed_text))
    if 'bag_of_pos_trigrams' in conf or all:
        features.extend(ngram_counts(3, 500, tokenized_text, pos=True)) # Tokenized for higher accuracy
    if 'bag_of_pos_bigrams' in conf or all:
        features.extend(ngram_counts(2, 100, tokenized_text, pos=True))
    if 'bag_of_pos_unigrams' in conf or all:
        features.extend(ngram_counts(1, 100, tokenized_text, pos=True))
    if 'characters_per_word' in conf or all:
        features.extend(characters_per_word(tokenized_text))
    if 'unique_words_ratio' in conf or all:
        features.extend(unique_words_ratio(tokenized_text))
    if 'words_per_sentence' in conf or all:
        features.extend(words_per_sentence(source_text)) # Source text b/c its necessary to sentence tokenize the essay
    if 'topic_model_scores' in conf or all:
        features.extend(topic_models(stemmed_text, 20))

    features = numpy.asarray(features).T.tolist() # transpose list of lists sow its dimensions are #instances x #features

    with open('features.csv', 'w') as ffile:
        fwriter = csv.writer(ffile)
        fwriter.writerow(header)
        fwriter.writerows(features)

    return features
