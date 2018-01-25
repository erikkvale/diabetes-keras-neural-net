from keras.models import Sequential
from keras.layers import Dense
import numpy as np


# Random seed for reproducibility
np.random.seed(7)

# Load dataset
dataset = np.loadtxt(r'./training_data/pima-indians-diabetes.csv', delimiter=",")

# Split dataset into X (input) and Y (output) variables
X = dataset[:, 0: 8]
Y = dataset[:, 8]

# Create model, add dense layers one by one specifying activation function
model = Sequential()
model.add(Dense(units=12, input_dim=8, activation='relu'))
model.add(Dense(units=15, activation='relu'))
model.add(Dense(units=8, activation='relu'))
model.add(Dense(units=10, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

# Compile model, adam gradient descent (optimized)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Call the function to fit to the data (training network)
model.fit(X, Y, epochs=1000, batch_size=10)

# Evaluate the model
scores = model.evaluate(X, Y)
print("{}: {}".format(model.metrics_names[1], scores[1]*100))


if __name__=='__main__':
    pass