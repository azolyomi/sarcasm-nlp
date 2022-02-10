#!/usr/bin/env python
# coding: utf-8

# # Read in the data #
import pandas as pd
import numpy as np
import os
import re # regex
import shutil
import string
import nltk

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras import Model, Input, Sequential
from tensorflow.keras.callbacks import *
from datetime import datetime
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import *
from tensorflow.keras.utils import plot_model

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer 
from nltk.tokenize import RegexpTokenizer

print('Tensorflow: ', tf.__version__)

class SarcasmModel:

    DEFAULT_PREDICTION_THRESHOLD = 0.6

    def __init__(self, path_to_build_folder="./", path_to_saved_model=None, train=False, epochs=30, save=True):
        self.path_to_build_folder = path_to_build_folder
        self.__instantiate_data()
        self.__cleanup_data()
        self.__gen_y_train_test()
        self.__tokenize_data()

        if (path_to_saved_model is not None):
            self.model = tf.keras.models.load_model(path_to_saved_model)
        else:
            self.__generate_embeddings()
            self.__prep_embeddings()
            self.__compile_model()
            if (train):
                self.train_model(epochs)
                if (save):
                    self.__save_model(f'{path_to_build_folder}saved_model/fake_news_v1')

    
    def __instantiate_data(self):
        self.test_dataset = pd.read_json(f'{self.path_to_build_folder}data/Sarcasm_Headlines_Dataset.json', lines=True)
        self.train_dataset = pd.read_json(f'{self.path_to_build_folder}data/Sarcasm_Headlines_Dataset_v2.json', lines=True)
        self.train_dataset = self.train_dataset.drop('article_link', axis=1)
    def __cleanup_data(self):
        self.test_dataset["headline"].apply(self.remove_contractions)
        self.train_dataset["headline"].apply(self.remove_contractions)
    def __gen_y_train_test(self):
        self.y_train = self.train_dataset["is_sarcastic"]
        self.y_test = self.test_dataset["is_sarcastic"]
        self.train_dataset.info()
    def __tokenize_data(self):
        self.t = Tokenizer()
        self.t.fit_on_texts(self.train_dataset["headline"])
        encoded_train = self.t.texts_to_sequences(self.train_dataset["headline"])
        encoded_test = self.t.texts_to_sequences(self.test_dataset["headline"])
        self.max_length = 25
        self.padded_train = pad_sequences(encoded_train, 
            maxlen = self.max_length, 
            padding = "post", 
            truncating = "post")
        self.padded_test = pad_sequences(encoded_test, 
            maxlen = self.max_length, 
            padding = "post", 
            truncating = "post")
        # print(self.padded_train.shape, self.padded_test.shape, type(self.padded_train))
        self.vocab_size = len(self.t.word_index) + 1

    def __generate_embeddings(self):
        path_to_glove_file = f'{self.path_to_build_folder}glove/glove.6B.100d.txt'
        self.embeddings_index = {}
        with open(path_to_glove_file) as f:
            for line in f:
                word, coefs = line.split(maxsplit=1)
                coefs = np.fromstring(coefs, "f", sep=" ")
                self.embeddings_index[word] = coefs
        print("Found %s word vectors." % len(self.embeddings_index))

    def __prep_embeddings(self):
        self.num_tokens = self.vocab_size + 2
        self.embedding_dim = 100
        hits = 0
        misses = 0
        # Prepare embedding matrix
        self.embedding_matrix = np.zeros((self.num_tokens, self.embedding_dim))
        for word, i in self.t.word_index.items():
            embedding_vector = self.embeddings_index.get(word)
            if embedding_vector is not None:
                # Words not found in embedding index will be all-zeros.
                # This includes the representation for "padding" and "OOV"
                self.embedding_matrix[i] = embedding_vector
                hits += 1
            else:
                misses += 1
        print("Converted %d words (%d misses)" % (hits, misses))

    def __compile_model(self):
        earlystop = EarlyStopping(monitor = "val_accuracy", 
                        patience = 7, 
                        verbose = 1,  
                        restore_best_weights = True, 
                        mode = 'max')

        reduce_lr = ReduceLROnPlateau(monitor = "val_accuracy", 
                        factor = .4642,
                        patience = 3,
                        verbose = 1, 
                        min_delta = 0.001,
                        mode = 'max')

        input = Input(shape = (self.max_length, ), name = "input")

        embedding = Embedding(input_dim = self.vocab_size + 2, 
                            output_dim = 100, 
                            weights = [self.embedding_matrix], 
                            trainable = False)(input)

        lstm = LSTM(32)(embedding)
        flatten = Flatten()(lstm)

        dense = Dense(16, activation = None, 
                    kernel_initializer = "he_uniform")(flatten)

        dropout = Dropout(.25)(dense)
        activation = Activation("relu")(dropout)
        output = Dense(2, activation = "softmax", name = "output")(activation)
        self.model = Model(inputs = input, outputs = output)
        self.model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])
        self.model.summary()

    def train_model(self, epochs=12):
        self.model.fit(self.padded_train, self.y_train, 
        validation_data = (self.padded_test, self.y_test), 
        epochs = epochs, 
        batch_size = 32)

    def __save_model(self, save_path):
        self.model.save(save_path)
    
    def predict_arr(self, input, threshold = DEFAULT_PREDICTION_THRESHOLD):
        standardized = self.standardize_map(input)
        prediction = self.model.predict(standardized)

        res = []
        for i in range(len(prediction)):
            if (prediction[i][1] >= threshold):
                res.append({"sarcastic": "true", "score": prediction[i][1].item()})
            else:
                res.append({"sarcastic": "false", "score": prediction[i][1].item()})
        return res

    def predict_singular(self, input, threshold = DEFAULT_PREDICTION_THRESHOLD):
        if not isinstance(input, str):
            return [{"error": "input must be a string"}]

        standardized = self.standardize_singular(input)
        prediction = self.model.predict(standardized)

        if (prediction[0][1] >= threshold):
            return {"sarcastic": "true", "score": prediction[0][1].item()}
        else:
            return {"sarcastic": "false", "score": prediction[0][1].item()}
    
    def predict(self, input, threshold = DEFAULT_PREDICTION_THRESHOLD):
        if isinstance(input, list):
            return self.predict_arr(input, threshold)
        elif isinstance(input, str):
            return self.predict_singular(input, threshold)
        else: return {"error": "input must be one of (string, list)"}

    def standardize(self, input_data):
        lowercase = tf.strings.lower(input_data)
        decontracted = self.remove_contractions(input_data)
        sequenced = self.t.texts_to_sequences([input_data])
        padded_sequenced = pad_sequences(
            sequenced,
            maxlen=self.max_length,
            padding = "post", 
            truncating = "post")
        return padded_sequenced

    def standardize_map(self, input_array):
        return map(self.standardize, input_array)
        
    def standardize_singular(self, input):
        return [self.standardize(input)]

    @classmethod
    def remove_contractions(self, sentence):
        sentence = re.sub(r"won\'t", "will not", sentence)
        sentence = re.sub(r"can\'t", "can not", sentence)
        sentence = re.sub(r"n\'t", " not", sentence)
        sentence = re.sub(r"\'re", " are", sentence)
        sentence = re.sub(r"\'s", " is", sentence)
        sentence = re.sub(r"\'d", " would", sentence)
        sentence = re.sub(r"\'ll", " will", sentence)
        sentence = re.sub(r"\'t", " not", sentence)
        sentence = re.sub(r"\'ve", " have", sentence)
        sentence = re.sub(r"\'m", " am", sentence)
        return sentence.lower()



