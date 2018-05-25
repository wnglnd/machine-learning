# -*- coding: utf-8 -*-
"""
Multi layer autoencoder
codings = 6 worked fine
codings = 5 made random predictions
codings = 4 worked fine
codings = 3 --||--
codings = 2 worked decent
codings = 1 worked decent, almost surprising
"""
import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.models import Model
from keras.layers import Input, Dense

N = 10
input_size = N
n_layer1 = N
n_layer2 = N
codings = 1
output_size = N
M = np.eye(N)

inputs = Input(shape = (input_size,))
layer1 = Dense(n_layer1, activation = 'relu')(inputs)
layer2 = Dense(n_layer2, activation = 'relu')(layer1)
h = Dense(codings, activation = 'relu')(layer2)
layer2_ = Dense(n_layer2, activation = 'relu')(h)
layer1_ = Dense(n_layer1, activation = 'relu')(layer2_)
outputs = Dense(output_size, activation = 'linear')(layer1_)

model = Model(input = inputs, output = outputs)
model.compile(loss = 'mse',
              optimizer = keras.optimizers.Adam(),
              metrics = ['accuracy'])

model.fit(M,
          M,
          epochs = 3000)

b = np.zeros((1,10))
b[0,1] = 1
y_hat = np.squeeze(model.predict(b))
plt.plot(y_hat, 'ro')
