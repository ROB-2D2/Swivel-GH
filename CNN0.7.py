'''This script loads pre-trained word embeddings (GloVe embeddings)
taken from: https://github.com/keras-team/keras/blob/master/examples/pretrained_word_embeddings.py
into a frozen Keras Embedding layer, and uses it to
train a text classification model on the 20 Newsgroup dataset
(classification of newsgroup messages into 20 different categories).
GloVe embedding data can be found at:
http://nlp.stanford.edu/data/glove.6B.zip
(source page: http://nlp.stanford.edu/projects/glove/)
20 Newsgroup data can be found at:
http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/news20.html
'''


"""
QUITE A FEW CHANGES HERE BECAUSE THINGS WERE NOT WORKING OUT. I HAVE CHANGED.
MAX_SEQUENCE_LENGTH. 
reviews_to_use 50000 -> 100000
convolution_neurons0 32 -> 64
convolution_neurons1 64 -> 96
convolution_neurons2 128 -> 128
convolution_window 5 -> 4
THE IDEA IS TO TRY SOMETHING QUITE DIFFERENT AND SEE HOW WE GO COMPARED TO THE
PREVIOUS WHICH WAS A LITTLE BETTER BUT STILL KINDA TERRIBLE.
"""


"""
INITIALISATION PHASE

"""


from __future__ import print_function
import os
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense
from keras.layers import Input
from keras.layers import GlobalMaxPooling1D 
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import Embedding
from keras.layers import Flatten
from keras.models import Model
from keras.models import Sequential
import json
import pickle

BASE_DIR = ''
GLOVE_DIR = os.path.join(BASE_DIR, 'C:\\Users\\dripd\\Desktop\\Zipline\\Swivel\\Stocks\\GloVe word embeddings')
MAX_SEQUENCE_LENGTH = 500
MAX_NUM_WORDS = 20000
VALIDATION_SPLIT = 0.2
reviews_to_use = 100000
convolution_neurons0 = 64
convolution_neurons1 = 96
convolution_neurons2 = 128
convolution_window = 4
pooling_size = 2
nr_of_epochs = 5
observations_per_batch = 64
EMBEDDING_DIM = [50, 100, 200, 300][3]
model_save_location = 'C:\\Users\\dripd\\Desktop\\Zipline\\Swivel\\Stocks\\Twitter\\models\\CNN0.7.h5'
tokeniser_name = 'PMtokeniser0.7'

"""
DATA PREPROCESSING PHASE
"""


# first, build index mapping words in the embeddings set
# to their embedding vector
print('Indexing word vectors.')

embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.' + str(EMBEDDING_DIM) + 'd.txt'), 
         encoding = "utf-8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

# second, prepare text samples and their labels
print('Processing text dataset')

#texts is just a list of the actual literal body of text
#labels_index i dont think we will need this
texts = []  # list of text samples
#labels_index = {}  # dictionary mapping label name to numeric id
labels = []  # list of label ids (1s and 0s for sentiment)

document_limit = reviews_to_use*0.5
happy_docs_loaded = 0
sad_docs_loaded = 0

i = 0
with open('C:\\Users\\dripd\\Desktop\\Zipline\\Swivel\\Sentiment Project\\data\\Amazon Agressive Dedupe Set\\aggressive_dedup.json', 'r') as json_file:
    for line in json_file:            
        json_instance = json.loads(line)
        if i % 1000 == 0:
            print(i)
        if json_instance["overall"] == 3:
            continue
        elif json_instance["overall"] > 3 and happy_docs_loaded < document_limit:
            labels.append(1)
            happy_docs_loaded += 1
        elif json_instance["overall"] < 3 and sad_docs_loaded < document_limit:
            labels.append(0)
            sad_docs_loaded += 1
        else:
            continue
        texts.append(json_instance["reviewText"])
        i += 1
        if i >= reviews_to_use:
            break

print('Loaded up %s happy texts.' % happy_docs_loaded)
print('Loaded up %s sad texts.' % sad_docs_loaded)

# finally, vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# saving a tokeniser
with open('tokenisers\\' + tokeniser_name + '.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
print('tokeniser saved...')

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
labels1 = labels
labels1 = to_categorical(np.asarray(labels))
labels = np.array(labels)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)


# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-num_validation_samples]
y_train = labels[:-num_validation_samples]
x_val = data[-num_validation_samples:]
y_val = labels[-num_validation_samples:]

print('Preparing embedding matrix.')

# prepare embedding matrix
num_words = min(MAX_NUM_WORDS, len(word_index))
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= MAX_NUM_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

"""
MODEL TRAINING PHASE
"""

# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed

# train a 1D convnet with global maxpooling

embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(convolution_neurons0, convolution_window, 
           activation='relu')(embedded_sequences)
x = MaxPooling1D(pooling_size)(x)
x = Conv1D(convolution_neurons1, convolution_window, activation='relu')(x)
x = MaxPooling1D(pooling_size)(x)
x = Conv1D(convolution_neurons2, convolution_window, activation='relu')(x)
x = GlobalMaxPooling1D()(x)
x = Dense(convolution_neurons2, activation='relu')(x)
preds = Dense(units = 1, activation='sigmoid')(x)

model = Model(sequence_input, preds)
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size = observations_per_batch,
          epochs = nr_of_epochs,
          validation_data=(x_val, y_val))

model.save(model_save_location)

"""
TWEET PREDICTION PHASE
"""

from keras.models import load_model

model_to_load = 'C:\\Users\\dripd\\Desktop\\Zipline\\Swivel\\Stocks\\Twitter\\models\\After dataset balancing\\VMCNN0.4.h5'

model = load_model(model_to_load)

save_file_name = 'Twitter\\TexasHumor.json'
number_of_tweets = 100

# loading a tokeniser
with open('C:\\Users\\dripd\\Desktop\\Zipline\\Swivel\\Stocks\\tokenisers\\VMtokenizer0.4.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

json_file = json.load(open(save_file_name))
json_file[3]['created_at']
json_file[3]['full_text']

#make a list of the text contained in each of the tweets:
list_of_prediction_text = []
for i in range(number_of_tweets):
    pred_text = json_file[i]['full_text']
    list_of_prediction_text.append(pred_text)

#list_of_prediction_text.append("Stocks pummeled as inflation shadow spooks bonds")
#list_of_prediction_text.append('i really dont know if this is any good')

pred_sequences = tokenizer.texts_to_sequences(list_of_prediction_text)
pred_data = pad_sequences(pred_sequences, maxlen=MAX_SEQUENCE_LENGTH)

pred_output = model.predict(pred_data)

final_predictions = []
for i in range(len(pred_output)):
    final_predictions.append(pred_output[i][0])

def how_many_happy_were_there():
    j = 0
    for i in range(len(final_predictions)):
        if final_predictions[i] >= 0.5:
            #print('happy')
            j += 1
    print(j)

"""
VISUALISATION AND NOUN EXTRACTION PHASE
"""

import nltk
import matplotlib.pyplot as plt

# POS tag all the texts
list_of_tagged_text = []
for i in range(len(list_of_prediction_text)):
    list_of_prediction_text[i] = list_of_prediction_text[i].split()
    list_of_tagged_text.append(nltk.pos_tag(list_of_prediction_text[i]))

# extract the nouns for each of the extracted tweets
noun_array = []
for i in range(len(list_of_tagged_text)):
    noun_list = []
    for tag_tuple in list_of_tagged_text[i]:
        if tag_tuple[1] == 'NN':
            #print(tag_tuple[0])
            noun_list.append(tag_tuple[0])
    noun_array.append(noun_list)

#create a dictionary of happy and sad words and their frequencies
happy_words = {}
sad_words = {}
noun_dict = {
        'happy' : happy_words,
        'sad' : sad_words
        }

# this function takes in the noun array and sentiment list created and turns
# them in to a dictionary where the nouns are associated with their frequency
# in happy or sad documents:
def convert_nouns_to_sentiment_dict(noun_array, sentiment_list):
    for i in range(number_of_tweets):
        for noun in noun_array[i]:
            if sentiment_list[i] >= 0.5:
                try:
                    noun_dict['happy'][noun] += 1
                except:
                    noun_dict['happy'][noun] = 1
            else:
                try:
                    noun_dict['sad'][noun] += 1
                except:
                    noun_dict['sad'][noun] = 1
    return noun_dict
            
noun_dict = convert_nouns_to_sentiment_dict(noun_array, final_predictions)

#remove nouns which didnt appear more than once
for sentiment in ['happy', 'sad']:
    for key, value in list(noun_dict[sentiment].items()):
        if value < 2:
            noun_dict[sentiment].pop(key, None)


plt.plot(final_predictions, color = 'red', label = 'Sentiment Over Time')
plt.title('Sentiment Tracker')
plt.xlabel('Time')
plt.ylabel('Sentiment')
plt.legend()
plt.show()




