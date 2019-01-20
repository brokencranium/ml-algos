import os
import re

import numpy as np
import pandas as pd
import pyprind
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

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
    return text.split()

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


if __name__ == "__main__":
    # input_data = getMoviesData()
    df = pd.read_csv('aclImdb/movie_data.csv', encoding='utf-8')
    df['review'] = df['review'].apply(preprocessor)

    print(df.head(3))
    print(df.shape)

    X_train = df.loc[:25000, 'review'].values
    y_train = df.loc[:25000, 'sentiment'].values
    X_test = df.loc[25000, 'review'].values
    y_test = df.loc[25000, 'sentiment'].values



    param_grid = [
        {
            'vect__ngram_range': [(1, 1)],
            'vect__stop_words': [stopwords, None],
            'vect__tokenizer': [tokenizer, tokenizer_porter],
            'clf__penalty': ['l1', 'l2'],
            'clf__C': [1.0, 10.0, 100.0]
        },
        {
            'vect__ngram_range': [(1, 1)],
            'vect__stop_words': [stopwords, None],
            'vect__tokenizer': [tokenizer, tokenizer_porter],
            'vect__use_idf': [False],
            'vect__norm': [None],
            'clf__penalty': ['l1', 'l2'],
            'clf__C': [1.0, 10.0, 100.0]
        }
    ]

    lr_tfidf = Pipeline([('vect', tf_idf),
                         ('clf',LogisticRegression(random_state=0))])
    gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid, scoring='accuracy',cv=5, verbose=1, n_jobs=1)
    gs_lr_tfidf.fit(X_train, y_train)

    print('Best parameter set: %s ' % gs_lr_tfidf.best_params_)
    print('CV Accuracy: %.3f' % gs_lr_tfidf.best_score_)
