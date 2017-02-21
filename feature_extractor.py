# extract features from list of text instances based on configuration set of features

import nltk
import numpy
import re
import time
import csv
import pickle
from nltk import ngrams
from collections import *
from gensim import corpora
from gensim.models.lsimodel import LsiModel

source_text = []
stemmed_text = []

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
