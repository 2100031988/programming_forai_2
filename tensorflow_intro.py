from tensorflow import keras
from tensorflow.keras import Dense
from tensorflow.keras import Sequential

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelBinarizer

import matplotlib.pyplot as plt
import numpy as np

iris = datasets.load_iris()
X = iris.data
y = iris.target 

X_columns = X.shape[1]

print("first 5 entries")
print(X[0:5])
print(X_columns)


# one hot encoding
# categorical to 0 and 1 (numerical format)

# color       is_orange       is_red      is_blue
# orange      1               0           0
# red         0               1           0
# blue        0               0           1

lb = LabelBinarizer()
y = lb.fit_transform(y)

y_classes = y.shape[1]

print(y_classes)
print(y[0:5])

X_train, y_train, X_test, y_test = train_test_split(
    X, y, test_size = 0.2, random_state = 42
)

# scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

# sequential
model = Sequential([
    Dense(10, activation = 'relu', input_shape = (X_columns, )),
    Dense(8, activation = 'relu'),
    Dense(y_classes, activation = 'sigmoid')
])

# compile
model.compile(
    optimizer = 'adam',
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)

# training and fit
history = model.fit(
    X_train, y_train,
    validation_data = (X_test, y_test),
    epochs = 50,
    batch_size = 5,
    verbose = 1
)

# evaluation
test_loss, test_acc = model.evaluate(X_test, y_test, verbose = 0)
print(f"Test accuracy: {test_acc:.4f}")

# plotting and visualization
plt.figure(figsize = (10, 4))

# accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label = 'Train Accuracy')
plt.plot(history.history['val_accuracy'], label = 'Validation Accuracy')
plt.title("Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

# loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label = 'Train Loss')
plt.plot(history.history['val_accuracy'], label = 'Validation Loss')
plt.title("Model Loss")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.show()


import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def create_nn(input_total, output_classes):
    model = Sequential([
        Dense(10, activation = 'relu', input_shape = (input_total, )),
        Dense(8, activation = 'relu'),
        Dense(output_classes, activation = 'sigmoid')
    ])
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model

models = {
    "Neural Networks" : create_nn(X_columns, y_classes),
    "kNN" : KNeighborsClassifier(n_neighbors=5)
}

results = []



