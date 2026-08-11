import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(8, activation = 'relu', input_shape = (8, )),
    Dense(4, activation = 'relu'),
    Dense(1, activation = 'sigmoid')
])

print(model.summary())

model.compile(
    optimizer = 'adam',
    loss = 'binary_crossentropy',
    metrics = "accuracy"
)

history = model.fit(
    # training data
    X_train,
    y_train,
    validation_data = (X_test, y_test),
    epochs = 50,
    batch_size = 5,
    verbose = 1
)








