import csv
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

dataset = []
targets = []

# Reads the csv dataset file and fills the arrays
with open('dataset.csv', newline='') as csv_file:
    reader = csv.DictReader(csv_file)
    for row in reader:
        dataset.append([row['R'], row['G'], row['B']])
        targets.append([row['IsItDark']])

# Convert string RGB values to normalized floats
dataset = np.array(dataset).astype('float32')
dataset /= 255

targets = np.array(targets).astype(np.float)

model = Sequential()
model.add(Dense(3, input_shape=(3,), activation='elu'))
model.add(Dense(2, input_shape=(3,), activation='elu'))
model.add(Dense(1, input_shape=(2,), activation='elu'))
model.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])

model.fit(dataset, targets, epochs=150, batch_size=128)
model.save('training.h5')