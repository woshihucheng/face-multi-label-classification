from __future__ import print_function
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import cPickle as pickle 
import numpy as np

num_classes = 4


# predicting data
pre = np.load('test-8.npy')

# convert list of 3D numpy array to 4D numpy array
pre = np.array(pre)


model = Sequential()
model.add(Conv2D(16, (5, 5), activation='relu', input_shape=(224, 184, 3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(48, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64, (2, 2), activation='relu'))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(num_classes, activation='sigmoid'))

# consider model ensemble
# TODO: search information about this usage
# model.load_weights('m1.h5')
# m_pre1 = model.predict(pre)

# model.load_weights('m2.h5')
# m_pre2 = model.predict(pre)

model.load_weights('model_final.h5')
m_pre3 = model.predict(pre)

# model.load_weights('m4.h5')
# m_pre4 = model.predict(pre)

# model.load_weights('m5.h5')
# m_pre5 = model.predict(pre)

# TODO: test
# ensemble = m_pre1 + m_pre2 + m_pre3 + m_pre4
# result = np.around(ensemble/4, 4)

result = np.round(m_pre3)
print(result)
# np.save('result.npy', result)
