import os
import re

import numpy as np
import pandas as pd
import pyprind
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

porter = PorterStemmer
tf_idf = TfidfTransformer(use_idf=True, norm='l2', smooth_idf=True)
count_vectorizer = CountVectorizer()
stopwords = stopwords.words('English')


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

def saveData(data, base_path='aclImdb'):
    np.random.seed(0)
    reindex_data = data.reindex(np.random.permutation(data.index))
    reindex_data.to_csv(base_path + '/movie_data.csv', index=False, encoding='utf-8')


def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = (re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', ''))
    return text


def tokenizer(text):
    return text.split()


def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split() if word not in stopwords]


def get_bag(tokens):
    return count_vectorizer.fit_transform(tokens), count_vectorizer.vocabulary


def get_tfidf(input):
    return tf_idf.fit_transform(input)


if __name__ == "__main__":
    # input_data = getMoviesData()
    df = pd.read_csv('aclImdb/movie_data.csv', encoding='utf-8')
    df['review'] = df['review'].apply(preprocessor)
    print(df.head(3))
    print(df.shape)

    _tokens = tokenizer_porter(df['review'])
    bag, vocabulary = get_bag(_tokens)

    np.set_printoptions(precision=2)
    print(get_tfidf(bag).toarray())
