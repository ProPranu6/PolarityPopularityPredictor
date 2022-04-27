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

class SentimentAnalyser:
  def __init__(self, df, word2vec):
    self.df = df
    self.word2vec = word2vec
    self.class_weights = None
    self.emd_wts = None
    self.sequence_len = 16
    self.sentiment_model = None

  def generate_class_weights(self):
    N = len(self.df)
    class_weights = {k : (N-v)/N  for k, v in self.df.label.value_counts().items()}
    self.class_weights = class_weights
    return

  def preprocess_text_data(self):
    global tokenizer

    data, word2vec = self.df, self.word2vec
    train_data, test_data = train_test_split(data, train_size=0.8, stratify=data['label'])
    tokenizer = Tokenizer(num_words=40000, oov_token= "<OOV>")
    tokenizer.fit_on_texts(data['text'])
    word_index = tokenizer.word_index
    training_sequences = tokenizer.texts_to_sequences(train_data['text'])
    training_padded = pad_sequences(training_sequences, maxlen=self.sequence_len, padding='post', truncating='post')
    testing_sequences = tokenizer.texts_to_sequences(test_data['text'])
    testing_padded = pad_sequences(testing_sequences, maxlen=self.sequence_len, padding='post', truncating='post')
    # convert lists into numpy arrays to make it work with TensorFlow 
    training_padded = np.array(training_padded)
    training_labels = np.array(train_data['label'])
    testing_padded = np.array(testing_padded)
    testing_labels = np.array(test_data['label'])

    embed_matrix = np.zeros((len(tokenizer.word_index)+1, 100))
    for word, i in tokenizer.word_index.items():
      try:
        embed_matrix[i] = word2vec.wv[word]
      except:
        pass

    self.emd_wts = embed_matrix

    return training_padded, testing_padded, training_labels, testing_labels
  
  def train(self, epochs=1):
    
    
    xtrain, xtest, ytrain, ytest = self.preprocess_text_data()
    embed = Embedding(self.emd_wts.shape[0], 100, weights=[self.emd_wts], input_length = (self.sequence_len,), trainable=True)


    Xinp = Input((self.sequence_len,))
    X = embed(Xinp)
    X = Flatten()(X)
    X = Dense(100, activation='relu')(X)
    X = Dense(25, activation='tanh')(X)
    X = Dense(1, activation='sigmoid')(X)


    self.generate_class_weights()
    model = Model(inputs=Xinp, outputs=X)
    model.compile(Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(xtrain, ytrain, epochs=epochs, class_weight=self.class_weights, batch_size=256)
    self.sentiment_model = model
  
  def predict_pipeline(self, headlines):

    predict_sequences = tokenizer.texts_to_sequences(headlines)
    predict_padded = pad_sequences(predict_sequences, maxlen=self.sequence_len, padding='post', truncating='post')
    predict_inp = np.array(predict_padded)
    
    raw_preds = self.sentiment_model.predict(predict_inp).reshape(-1)
    preds = (raw_preds >= 0.5).astype(int)
    labels = np.array(['Negative', 'Positive'])
    
    return labels[preds], raw_preds
  






