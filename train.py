import os
import numpy as np
import tensorflow as tf
import trimesh.sample
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
import pickle

import network
import utils


# If using a GPU keep these lines to avoid CUDNN errors
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

DATA_DIR = "ModelNet10/"     # <- Set this path correctly


num_points_per_cloud = 4096
# train_pc, test_pc, train_labels, test_labels, class_ids = utils.create_point_cloud_dataset(DATA_DIR, num_points_per_cloud)

# pickle.dump(train_pc, open("trainpc.pkl", "wb"))
# pickle.dump(test_pc, open("testpc.pkl", "wb"))
# pickle.dump(train_labels, open("trainlabels.pkl", "wb"))
# pickle.dump(test_labels, open("testlabels.pkl", "wb"))
# pickle.dump(class_ids, open("class_ids.pkl", "wb"))

# load the data from pickle files if already present
train_pc = pickle.load(open("trainpc.pkl", "rb"))
train_labels = pickle.load(open("trainlabels.pkl", "rb"))
test_pc = pickle.load(open("testpc.pkl", "rb"))
test_labels = pickle.load(open("testlabels.pkl", "rb"))
class_ids = pickle.load(open("class_ids.pkl", "rb"))

# Create tensorflow data loaders from the numpy arrays
# print(test_pc.shape)
# print(test_labels.shape)
train_dataset = tf.data.Dataset.from_tensor_slices((train_pc, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_pc, test_labels))

batch_size = 32
# print(len(train_pc))
train_dataset = train_dataset.shuffle(len(train_pc)).map(utils.add_noise_and_shuffle).batch(batch_size)
test_dataset = test_dataset.shuffle(len(test_pc)).batch(batch_size)

inputs = keras.Input(shape=(num_points_per_cloud, 3))
outputs = network.pointnet_classifier(inputs, num_classes=10)
# outputs = network.pointnet_segmenter(inputs, train_labels)

# build the network and visualize its architecture
model = keras.Model(inputs=inputs, outputs=outputs, name="pointnet")
model.summary()

# 2. Set the loss function, optimizer and metrics to print
model.compile(
    loss=keras.losses.CategoricalCrossentropy(),     # <- choose a suitable loss function
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),      # <- you may modify this if you like
    metrics=["accuracy"],    # <- choose a suitable metric, https://www.tensorflow.org/api_docs/python/tf/keras/metrics
)

# train the network
num_epochs = 1      # <- change this value as needed
model.fit(train_dataset, epochs=num_epochs, validation_data=test_dataset)

model.save('classifier_model')




# visualize results
# data = test_dataset.take(1)
# point_clouds, labels = list(data)[0]  # this is one batch of data

# # predict labels using the model
# preds = model.predict(point_clouds)
# preds = tf.math.argmax(preds, -1)

# 3. Display some clouds using matplotlib scatter plot along with true and predicted labels

# 4. Display a confusion matrix