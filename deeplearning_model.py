from os.path import exists

import numpy as np
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv1D, Flatten, Dense

from sklearn.model_selection import train_test_split

DATA_PATH = 'C:\\Users\\Mohamed\\PycharmProjects\\DATA'
labels_train = np.load('10percentOfy_train.npy')
labels_test = np.load(DATA_PATH+'/Test/y_test.npy')

Train_set_size = len(labels_train)
Test_set_size = len(labels_test)

actions = ['sweeping', 'gargling', 'opening cupboard', 'washing hands', 'eating', 'writing', 'wiping',
           'drinking','opening microwave oven', 'Throwing trash']


def create_model(input_shape):
    model = Sequential()
    model.add(Conv1D(64, 5))#, data_format="channels_first"))
    model.add(Conv1D(32, 5))#, data_format="channels_first"))
    model.add(Conv1D(16, 5))#, data_format="channels_first"))
    model.add(Conv1D(8, 5))#, data_format="channels_first"))
    model.add(Flatten())
    model.add(Dense(1000))
    model.add(Dense(10, activation = 'softmax'))
    model.compile(optimizer='sgd', loss='mse', metrics=["accuracy"])
    model.build(input_shape)
    model.summary()
    return model

def load_train_embeddings(dim, length, walks):
    embeddings_path = DATA_PATH + '/embeddings'
    X_train = []
    for i in range(Train_set_size):
        file = embeddings_path + '/train/emb_' + str(dim) + 'dim_' + str(length) + 'len_' + str(walks) + 'walks_' + str(i) + '.npy'
        X_train.append(np.load(file))
    return X_train

def load_test_embeddings(dim, length, walks):
    embeddings_path = DATA_PATH + '/embeddings'
    X_test = []
    for i in range(Test_set_size):
        file = embeddings_path + '/test/emb_' + str(dim) + 'dim_' + str(length) + 'len_' + str(walks) + 'walks_' + str(i) + '.npy'
        X_test.append(np.load(file))
    return X_test

from keras.utils.np_utils import to_categorical

def to_category(y_train):
    y_tr = [actions.index(action) for action in y_train]
    y_tr = to_categorical(y_tr)
    return y_tr

# models_path = "models"

"""
y_train, y_test = to_category(labels_train, labels_test)
train_emb = 'embeddings/train/'
test_emb = 'embeddings/test/'

X_train2 = np.load(train_emb+'train_embedding_dim2.npy')
X_train4 = np.load(train_emb+'train_embedding_dim4.npy')
X_train8 = np.load(train_emb+'train_embedding_dim8.npy')

X_test8 = np.load(test_emb+'test_embedding_dim8.npy')
X_test4 = np.load(test_emb+'test_embedding_dim4.npy')
X_test2 = np.load(test_emb+'test_embedding_dim2.npy')
"""


# X_val, y_val, X_test, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)


"""
x_path = "embeddings/train/train_embedding_dim8.npy"
X = np.load(x_path)
X_train, X_test, y_train, y_test = train_test_split(X, y_train, test_size = 0.1, random_state=123)
model, history = fit_model(8, X_train, y_train, X_test, y_test)
model.evaluate(X_test, y_test)

fig, axs = plt.subplots(2, 1, figsize=(15,15))
axs[0].plot(history.history['loss'])
axs[0].plot(history.history['val_loss'])
axs[0].title.set_text('Training Loss vs Validation Loss ')
axs[0].legend(['Train', 'Val'])
axs[1].plot(history.history['accuracy'])
axs[1].plot(history.history['val_accuracy'])
axs[1].title.set_text('Training Accuracy vs Validation Accuracy ')
axs[1].legend(['Train', 'Val'])
plt.show()
"""
#input_shape = (None,1000,32)
#model = create_model(input_shape)

def fit_model():
    labels = np.load(DATA_PATH+"/Train/y_train.npy")
    y = to_category(labels)
    X = np.load("embeddings/train/train_embedding_dim8.npy")
    model = create_model((None,1000,8))
    history = model.fit(X,y, epochs=100, validation_split=0.2)
    return model, history

def viz_accuracy(history):
    fig, axs = plt.subplots(2, 1, figsize=(15,15))
    axs[0].plot(history.history['loss'])
    axs[0].plot(history.history['val_loss'])
    axs[0].title.set_text('Training Loss vs Validation Loss dim = 8 tout le dataset')
    axs[0].legend(['Train', 'Val'])
    axs[1].plot(history.history['accuracy'])
    axs[1].plot(history.history['val_accuracy'])
    axs[1].title.set_text('Training Accuracy vs Validation Accuracy dim = 8 tout le dataset')
    axs[1].legend(['Train', 'Val'])
    plt.show()

model, history = fit_model()

viz_accuracy(history)