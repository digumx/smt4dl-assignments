"""
Script to train a network using data generated by datagen and using tensorflow.keras
"""

import os.path

from tensorflow.keras import models, layers, losses

from datagen import *


data = []
num = 10000

# Check if datafile exists
if os.path.isfile("./out/data.val"):
    with open("./out/data.val") as f:
        data = eval(f.read())
        num = len(data)
else:
    with open("./out/data.val", "w") as f:
        data = gen_data(num)
        f.write(str(data))

# Create a simple DNN model
dnn = models.Sequential([
        layers.Input(shape = (36,)),
        layers.Dense(20, activation='relu'),
        layers.Dense(10, activation='relu'),
        layers.Dense(2, activation='relu'),])

# We do not use the softmax normalization in the DNN itself. Instead, we bake a normalization into
# the loss function by using the BinaryCrossentropy function with the from_logits flag set to True.
# This will effectively apply softmax while calculating loss, but the model itself does not apply
# softmax to the output.
dnn.compile(loss = losses.BinaryCrossentropy(from_logits = True), metrics = ['accuracy'])

# Split training and testing data, prepare data
train_x = list(map(lambda x : [1 if c else 0 for c in x], [p[0] for p in data[:int(0.6*num)]]))
train_y = [([0, 1] if p[1] else [1, 0]) for p in data[:int(0.6*num)]]
test_x = list(map(lambda x : [1 if c else 0 for c in x], [p[0] for p in data[int(0.6*num):]]))
test_y = [([0, 1] if p[1] else [1, 0]) for p in data[int(0.6*num):]]

# Train model
dnn.fit(train_x, train_y, epochs=80)

# Test model
dnn.evaluate(test_x, test_y, verbose=2)
