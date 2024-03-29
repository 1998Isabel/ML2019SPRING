#!/usr/bin/env python
import sys, os
import numpy as np
from keras.models import load_model
from keras.utils import plot_model
import csv

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if len(directory) == 0: return
    if not os.path.exists(directory):
        os.makedirs(directory)

# Parameter
height = width = 48
num_classes = 7
input_shape = (height, width, 1)
model_name = sys.argv[3]
# model_name = 'Drive/ML_hw/hw3/model/weights.050-0.70800.h5'

# Read the test data
with open(sys.argv[1], "r+") as f:
    line = f.read().strip().replace(',', ' ').split('\n')[1:]
    raw_data = ' '.join(line)
    length = width*height+1 #1 is for label
    data = np.array(raw_data.split()).astype('float').reshape(-1, length)
    X = data[:, 1:]
    X /= 255

# Load model
model = load_model(model_name)

# Plot model
# plot_model(model,to_file='cnn_model.png')

# Predict the test data
X = X.reshape(X.shape[0], height, width, 1)
ans = model.predict_classes(X)
ans = list(ans)

# Write prediction
## check the folder of out.csv is exist; otherwise, make it
ensure_dir(sys.argv[2])

result = []
for index, value in enumerate(ans):
    result.append("{0},{1}".format(index, value))

with open(sys.argv[2], 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['id','label'])
        for i in range(len(ans)):
                # id_name = 'id_' + str(i)
                writer.writerow([i, ans[i]])