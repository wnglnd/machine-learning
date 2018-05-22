# -*- coding: utf-8 -*-
"""
Vanilla autoencoder with shape
input = 10
hidden layer = 8
output = 10
"""
import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.models import Model
from keras.layers import Input, Dense

N = 10
input_size = N
hidden_size = 8
output_size = N
M = np.eye(N)

inputs = Input(shape = (input_size,))
h = Dense(hidden_size, activation = 'relu')(inputs)
outputs = Dense(output_size, activation = 'linear')(h)

model = Model(input = inputs, output = outputs)
model.compile(loss = 'mae',
              optimizer = keras.optimizers.Adam(),
              metrics = ['accuracy'])

model.fit(M,
          M,
          epochs = 1000)

b = np.zeros((1,10))
b[0,1] = 1
y_hat = np.squeeze(model.predict(b))
plt.plot(y_hat, 'ro')
