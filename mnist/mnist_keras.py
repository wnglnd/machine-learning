# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 14:31:56 2017

"""
from __future__ import print_function
import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization, Activation
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt


def split_into_train_dev_test(X, train_size = 0.98):
    """
    Split the labeled data into train, dev and test sets. 
    """
    dev_test_size = 1 - train_size
    rs  = ShuffleSplit(n_splits = 1, test_size = dev_test_size)
    for train_index, test_index in rs.split(X):
        X_train     = X.iloc[train_index]
        X_test_dev  = X.iloc[test_index]
        
    rs  = ShuffleSplit(n_splits = 1, test_size = 0.5)
    for train_index, test_index in rs.split(X_test_dev):
        X_dev   = X_test_dev.iloc[train_index]
        X_test  = X_test_dev.iloc[test_index]
        
    return X_train, X_dev, X_test


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert inputs.shape[0] == targets.shape[0]
    m   = inputs.shape[0]
    if shuffle:
        indices = np.arange(m)
        np.random.shuffle(indices)
    for start_idx in range(0, m - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]
     
        
def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
	
    Arguments:
    X -- input data, of shape (input size, number of examples) (m, Hi, Wi, Ci)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples) (m, n_y)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.
 	
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
 	
    m = X.shape[0]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
  	
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:,:,:]
    shuffled_Y = Y[permutation,:]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = int(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:,:,:]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:,:,:]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches

def shuffle_x_y(X, Y, seed = 0):
    m   = X.shape[0]
    np.random.seed(seed)
    permutation = list(np.random.permutation(m))
    shuffled_X  = X[permutation,:,:,:]
    shuffled_Y  = Y[permutation,:]
    
    return shuffled_X, shuffled_Y

def shuffled_train_test_index(m, train_size = 0.94, seed = 0):
    np.random.seed(seed)
    permutation = np.random.permutation(m)
    m_train     = int( m * train_size)
    m_dev       = int( (m - m_train) / 2)
    
    train_index = range(0,m_train)
    train_index = permutation[train_index]
    dev_index   = range(m_train,(m_train + m_dev))
    dev_index   = permutation[dev_index]
    test_index  = range((m_train + m_dev),m)
    test_index  = permutation[test_index]
    
    return train_index, dev_index, test_index
    


train_raw   = pd.read_csv('../../data/mnist_kaggle/train.csv')
test_raw    = pd.read_csv('../../data/mnist_kaggle/test.csv')

img_w               = 28
img_h               = 28
input_shape         = (img_h, img_w, 1)
X_train             = train_raw.drop('label', axis=1).as_matrix().astype(np.float32)
Y_train             = train_raw['label'].copy().as_matrix().reshape(X_train.shape[0],1).astype(np.int32)
X_sub               = test_raw.as_matrix().astype(np.float32)

stdScaler       = StandardScaler()
X_train_scaled  = stdScaler.fit_transform(X_train).astype(np.float32)
X_sub_scaled    = stdScaler.transform(X_sub).astype(np.float32)

X_train_scaled  = X_train_scaled.reshape(X_train_scaled.shape[0], img_h, img_w, 1)
X_sub_scaled    = X_sub_scaled.reshape(X_sub_scaled.shape[0], img_h, img_w, 1)
train_index, dev_index, test_index = shuffled_train_test_index(X_train_scaled.shape[0])
Y_train         = keras.utils.to_categorical(Y_train, num_classes = 10)


model = Sequential()
model.add( Conv2D(filters = 64, kernel_size = (3, 3), padding = 'same', input_shape = input_shape, name = 'Conv1'))
model.add( BatchNormalization(name = 'BN1'))
model.add( Activation('relu'))

model.add( Conv2D(filters = 32, kernel_size = (3, 3), padding = 'valid', name = 'Conv2'))
model.add( BatchNormalization(name = 'BN2'))
model.add( Activation('relu'))
model.add( MaxPool2D((3, 3), strides = (2, 2), name = 'MaxPool2'))

model.add( Conv2D(filters = 30, kernel_size = (3, 3), padding = 'valid', name = 'Conv3'))
model.add( BatchNormalization(name = 'BN3'))
model.add( Activation('relu'))
model.add( MaxPool2D((3, 3), strides = (2, 2), name = 'MaxPool3'))

model.add( Flatten())
model.add( Dropout(0.4, name = 'Dropout4'))
model.add( Dense(40, activation = 'relu', name = 'FC4'))
model.add( BatchNormalization(name = 'BN4'))

model.add( Dropout(0.4, name = 'Dropout5'))
model.add( Dense(10, activation = 'softmax', name = 'FC5'))

model.compile(loss = keras.losses.categorical_crossentropy,
              optimizer = keras.optimizers.Adam(),
              metrics = ['accuracy'])

model.fit(X_train_scaled[train_index],
          Y_train[train_index],
          epochs = 12,
          verbose = 1,
          validation_data = (X_train_scaled[dev_index], Y_train[dev_index]))
