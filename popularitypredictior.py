
#import libraries and modules
from google.colab import files
import io
import pandas as pd
#Snorkel
from snorkel.labeling import LabelingFunction
import re
from snorkel.preprocess import preprocessor
from textblob import TextBlob
from snorkel.labeling import PandasLFApplier
from snorkel.labeling.model import LabelModel
from snorkel.labeling import LFAnalysis
from snorkel.labeling import filter_unlabeled_dataframe
from snorkel.labeling import labeling_function
#NLP packages
import spacy
from nltk.corpus import stopwords
import string
import nltk
import nltk.tokenize
punc = string.punctuation
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
#Supervised learning
from tqdm import tqdm_notebook as tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
##Deep learning libraries and APIs
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Dense, Embedding, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import gensim.downloader as api
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

class PopularityPredictior:
  def __init__(self, df):
    self.encs = []
    self.df, self.dataset = self.reset_preprocess_df(df)
    self.popmodel = None
    
    pass
  
  def reset_preprocess_df(self, df, for_predict=False):
    try:
      df.drop(['subsection', 'uniqueID'], axis=1, inplace=True)
    except:
      pass
    try: 
      df = df.dropna()
    except:
      pass

    #segragate times to before_eve(0) and after_eve(1)
    times = list(map(int, re.findall('([0-9]{2}):[0-9]{2}:[0-9]{2}', " ".join(list(df['pub_date'])))))
    df['times'] = times
    dic = {}
    for k in range(0, 24):
      if k<15:
        dic[k]=0
      else:
        dic[k]=1 
    df['times'] = df['times'].replace(dic)

    #added gaussian word count using boxcox transformation
    lam =0.5
    data = ((df['word_count'])**lam - 1)/lam
    df['normalised_word_count'] = data

    #tokenizing the keywords
    df['keywords'] = df['keywords'].apply(lambda x: " ".join(eval(x)))

    if for_predict:
      te = self.encs[0]
    else:
      te = Tokenizer()
      te.fit_on_texts(df['keywords'])
    
    sequences = te.texts_to_sequences(df['keywords'])
    paded_sequences = pad_sequences(sequences, maxlen=16, padding='post', truncating='post')
    df['keywords'] = [str(seq) for seq in paded_sequences]

    #Categorize newsdesk, section and material columns
    if for_predict:
      c1, c2, c3 = self.encs[1:]
    else:
      c1, c2, c3 = LabelEncoder(), LabelEncoder(), LabelEncoder() #pd.Categorical(df['newsdesk']), pd.Categorical(df['section']), pd.Categorical(df['material'])
      c1.fit(df['newsdesk']), c2.fit(df['section']), c3.fit(df['material'])
    
    df['newsdesk'], df['section'], df['material'] = c1.transform(df['newsdesk']), c2.transform( df['section']), c3.transform(df['material'])

    #store all the encoders
    if not(for_predict):
      self.encs += [te, c1, c2, c3]

    #drop remaining cols
    try:
      df.drop(['abstract', 'pub_date'], inplace=True, axis=1)
    except:
      pass

    if for_predict:
      dataframe = df[['times', 'normalised_word_count', 'newsdesk', 'section', 'material', 'n_comments']]
      x = np.array(dataframe)
      x = np.concatenate([x, np.array(paded_sequences)], axis=-1)
      return x
    else:
      
      #generate xtrain, ytrain, xtest, ytest
      dataframe = df[['times', 'normalised_word_count', 'newsdesk', 'section', 'material', 'n_comments']]
      x = np.array(dataframe)
      x = np.concatenate([x, np.array(paded_sequences)], axis=-1)
      y = np.array(df['is_popular'])

      xtrain, xtest, ytrain, ytest = train_test_split(x, y, train_size=0.8, stratify=y)

      return df, [xtrain, xtest, ytrain, ytest]

  def train(self):
    xtrain, xtest, ytrain, ytest = self.dataset
    dtc = DecisionTreeClassifier()
    dtc.fit(xtrain, ytrain)
    self.popmodel = dtc

    return dtc

  def __call__(self):
    return

  def predict(self, xdf, preprocess=True):

    if preprocess:
      x = self.reset_preprocess_df(xdf, for_predict=True)
    else:
      x = xdf

    return self.popmodel.predict(x)
    






