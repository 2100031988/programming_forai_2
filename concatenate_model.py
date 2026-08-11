# used when we have different types of input layers and they have categories

from tensorflow.keras.layers import Dense, Concatenate
from tensorflow.keras import Model, Input

input_a = Input(shape = (2, ))
input_b = Input(shape = (3, ))

x = Dense(3, activation = 'relu')(input_a)
y = Dense(5, activation = 'relu')(input_b)

concat = Concatenate()[x, y]
output = Dense(7, activation = 'softmax')(concat)
model = Model(inputs=[input_a, input_b], outputs=output)