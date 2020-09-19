#Installing and importing libraries

!pip install tensorflow
import tensorflow as tf
import numpy as np
from tensorflow import keras
print(tf.__version__)

#Representing Data in the form of arrays

xs = np.array([1, 2, 3, 4, 5, 6], dtype='float')
ys = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5], dtype='float')

#Neural Network

model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

#Compile

model.compile(optimizer='sgd', loss='mean_squared_error')

#Traing the model

model.fit(xs, ys, epochs=500)

#Predicting with the trained model

print(model.predict([7.0]))
