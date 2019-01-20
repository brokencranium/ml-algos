import os
import pickle
import re

import numpy as np
import pandas as pd
import pyprind
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.linear_model import SGDClassifier

porter = PorterStemmer
count_vectorizer = CountVectorizer()
stopwords = stopwords.words('English')
tf_idf = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None)


# Do not add leading slash to the path
def getMoviesData(base_path='aclImdb'):
    labels = {'pos': 1, 'neg': 0}
    pbar = pyprind.ProgBar(50000)
    df = pd.DataFrame()

    for s in ('test', 'train'):
        for l in ('pos', 'neg'):
            path = os.path.join(base_path, s, l)
            print(path)

            for file in sorted(os.listdir(path)):
                with open(os.path.join(path, file), 'r', encoding='utf-8') as infile:
                    txt = infile.read()
                    df = df.append([[txt, labels[l]]], ignore_index=True)
                    pbar.update()
    return df


# def download_nltk():
# import nltk
# nltk.download('stopwords')

def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = (re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', ''))
    return text


def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    tokenized = [w for w in text.split() if w not in stopwords]
    return tokenized


def stream_docs(path):
    with open(path, 'r', encoding='utf-8') as csv:
        next(csv)  # skip header
        for line in csv:
            text, label = line[:-3], int(line[-2])
            yield text, label


def get_minibatch(doc_stream, size):
    docs, y = [], []
    try:
        for _ in range(size):
            text, label = next(doc_stream)
            docs.append(text)
            y.append(label)

    except StopIteration:
        return None, None
    return docs, y


def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split() if word not in stopwords]


def saveData(data, base_path='aclImdb'):
    np.random.seed(0)
    reindex_data = data.reindex(np.random.permutation(data.index))
    reindex_data.to_csv('aclImdb/movie_data.csv', index=False, encoding='utf-8')


def get_bag(input):
    return count_vectorizer.fit_transform(input), count_vectorizer.vocabulary


def get_tfidf(input):
    return tf_idf.fit_transform(input)


def pickle_it(input, name, base_path='aclImdb'):
    dest = os.path.join(base_path, 'pickled/' + name + '.pkl')
    pickle.dump(input, open(dest, 'wb'), protocol=4)


if __name__ == "__main__":
    # input_data = getMoviesData()
    # Use hashing vectorizer since count vectorizer requires holding all the data in memory
    vect = HashingVectorizer(decode_error='ignore', n_features=2 ** 21, preprocessor=None,
                             tokenizer=tokenizer)
    clf = SGDClassifier(loss='log', random_state=1, max_iter=1, tol=1e-3)
    doc_stream = stream_docs(path='aclImdb/movie_data.csv')
    pbar = pyprind.ProgBar(45)
    classes = np.array([0, 1])

    for _ in range(45):
        X_train, y_train = get_minibatch(doc_stream, size=1000)
        if not X_train:
            break
        X_train = vect.transform(X_train)
        clf.partial_fit(X_train, y_train, classes=classes)
        pbar.update()

    X_test, y_test = get_minibatch(doc_stream, size=5000)
    X_test = vect.transform(X_test)
    print('Accuracy: %.3f' % clf.score(X_test, y_test))

    clf = clf.partial_fit(X_test, y_test)

    pickle_it(clf, 'clf_sgd')
    pickle_it(stopwords, 'dat_s_words')
