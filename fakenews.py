#!/usr/bin/env python
# coding: utf-8

# # Read in the data #

# In[27]:


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

print(tf.__version__)


# In[28]:


test_dataset = pd.read_json('./data/Sarcasm_Headlines_Dataset.json', lines=True)
test_dataset.head()


# In[29]:


train_dataset = pd.read_json('./data/Sarcasm_Headlines_Dataset_v2.json', lines=True)
train_dataset.info()


# # Basic data cleanup and column removal #

# In[30]:


train_dataset = train_dataset.drop('article_link', axis=1)


# In[31]:


train_dataset.info()


# # Pre-processing #
# 1. contractions
# 2. stop words
# 3. lowercase
# 4. stemming
# 5. tokenize
# 
# Q: Do we need to do all of these?

# ### Contractions ###

# In[32]:


def remove_contractions(sentence):
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


# In[33]:


train_dataset["headline"].apply(remove_contractions)
test_dataset["headline"].apply(remove_contractions)


# In[34]:


y_train = train_dataset["is_sarcastic"]
y_test = test_dataset["is_sarcastic"]


# In[35]:


train_dataset.info()


# ### Stop Words ###

# In[36]:


nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer 
from nltk.tokenize import RegexpTokenizer


# #### One way of doing it: ####

# In[38]:


print(train_dataset["headline"][0])


# ## Tokenize ##

# In[39]:


t = Tokenizer()
t.fit_on_texts(train_dataset["headline"])

encoded_train = t.texts_to_sequences(train_dataset["headline"])
encoded_test = t.texts_to_sequences(test_dataset["headline"])

max_length = 25

padded_train = pad_sequences(encoded_train, 
    maxlen = max_length, 
    padding = "post", 
    truncating = "post")

padded_test = pad_sequences(encoded_test, 
    maxlen = max_length, 
    padding = "post", 
    truncating = "post")

print(padded_train.shape, padded_test.shape, type(padded_train))

vocab_size = len(t.word_index) + 1
vocab_size

# In[41]:
path_to_glove_file = "./glove/glove.6B.100d.txt"

embeddings_index = {}
with open(path_to_glove_file) as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, "f", sep=" ")
        embeddings_index[word] = coefs

print("Found %s word vectors." % len(embeddings_index))

# In[42]:
num_tokens = vocab_size + 2
embedding_dim = 100
hits = 0
misses = 0

# Prepare embedding matrix
embedding_matrix = np.zeros((num_tokens, embedding_dim))
for word, i in t.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # Words not found in embedding index will be all-zeros.
        # This includes the representation for "padding" and "OOV"
        embedding_matrix[i] = embedding_vector
        hits += 1
    else:
        misses += 1
print("Converted %d words (%d misses)" % (hits, misses))


# In[43]:
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


# In[44]:
input = Input(shape = (max_length, ), name = "input")

embedding = Embedding(input_dim = vocab_size + 2, 
                      output_dim = 100, 
                      weights = [embedding_matrix], 
                      trainable = False)(input)

lstm = LSTM(32)(embedding)
flatten = Flatten()(lstm)

dense = Dense(16, activation = None, 
              kernel_initializer = "he_uniform")(flatten)

dropout = Dropout(.25)(dense)
activation = Activation("relu")(dropout)
output = Dense(2, activation = "softmax", name = "output")(activation)
model = Model(inputs = input, outputs = output)

model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])

model.summary()


# In[45]:
model.fit(padded_train, y_train, 
        validation_data = (padded_test, y_test), 
        epochs = 12, 
        batch_size = 32)


# In[55]:
def standardize_singular(input_data):
    lowercase = tf.strings.lower(input_data)
    decontracted = remove_contractions(input_data)
    sequenced = t.texts_to_sequences([input_data])
    padded_sequenced = pad_sequences(
        sequenced,
        maxlen=max_length,
        padding = "post", 
        truncating = "post")
    return padded_sequenced

def standardize_map(input_array):
    return map(standardize_singular, input_array)
    


# In[56]:
examples = [
    "Breakthrough Procedure Allows Surgeons To Transplant Pig Rib Directly Into Human Mouth",
    "Jan. 6 Committee Seeks Interview With Kevin McCarthy",
    "CDC Shortens COVID Isolation Guidelines to One Pump Up Song on Way to Work",
    "Report: Snickers Basically Protein Bar",
    "Crappy Music Has Helped Moron Through Hardest Times In His Pointless Life",
    "Is Eminem Leaving The Music Industry?",
    "Breaking News: Eminem Scared Out Of Music Industry By A Polar Bear",
    "Chancellor Yang praised for bold indecision",
    "Breaking News: UCSB shuts down permanently after Yang decides its too annoying to be a chancellor",
    "Munger Hall described as \"Absolutely Stunning\" by UCSB Chancellor",

]


standardized = standardize_map(examples)

prediction = model.predict(standardized)

for i in range(len(prediction)):
    if (prediction[i][1] >= 0.6):
        print("I'm pretty sure that's sarcastic...", prediction[i][1])
    else:
        print("I buy it!", prediction[i][1])




