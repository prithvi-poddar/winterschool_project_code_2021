import os
import numpy as np
import tensorflow as tf
import trimesh.sample
from tensorflow import keras
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from tensorflow.keras import layers
from matplotlib import pyplot as plt
import pickle
import tensorboard
from datetime import datetime

# import network
import reduced_network as network
import utils
import provider

# If using a GPU keep these lines to avoid CUDNN errors
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

DATA_DIR = "ModelNet10/"     # <- Set this path correctly


num_points_per_cloud = 1024
# train_pc, test_pc, train_labels, test_labels, class_ids = utils.create_point_cloud_dataset(DATA_DIR, num_points_per_cloud)

# # print(train_pc.shape)
# train_pc = provider.normalize_pc(train_pc)
# # print(train_pc.shape)
# test_pc = provider.normalize_pc(test_pc)

# pickle.dump(train_pc, open("trainpc.pkl", "wb"))
# pickle.dump(test_pc, open("testpc.pkl", "wb"))
# pickle.dump(train_labels, open("trainlabels.pkl", "wb"))
# pickle.dump(test_labels, open("testlabels.pkl", "wb"))
# pickle.dump(class_ids, open("class_ids.pkl", "wb"))

# utils.visualize_cloud(train_pc[0])


# load the data from pickle files if already present
train_pc = pickle.load(open("trainpc.pkl", "rb"))
train_labels = pickle.load(open("trainlabels.pkl", "rb"))
test_pc = pickle.load(open("testpc.pkl", "rb"))
test_labels = pickle.load(open("testlabels.pkl", "rb"))
class_ids = pickle.load(open("class_ids.pkl", "rb"))

train_pc = provider.rotate_point_cloud(train_pc)
# test_pc = provider.rotate_point_cloud(test_pc)

# Create tensorflow data loaders from the numpy arrays

train_dataset = tf.data.Dataset.from_tensor_slices((train_pc, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_pc, test_labels))

batch_size = 32

train_dataset = train_dataset.shuffle(len(train_pc)).map(utils.add_noise_and_shuffle).batch(batch_size)
test_dataset = test_dataset.shuffle(len(test_pc)).batch(batch_size)

inputs = keras.Input(shape=(num_points_per_cloud, 3))
outputs = network.pointnet_classifier(inputs, num_classes=10)
# outputs = network.pointnet_segmenter(inputs, train_labels)

# build the network and visualize its architecture
model = keras.Model(inputs=inputs, outputs=outputs, name="pointnet")
model.summary()

logdir = "logs/classifier_3/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=100)
model_checkpoint = ModelCheckpoint('classifier_model_3_best.h5', monitor='val_accuracy', mode='max', save_best_only=True)


# 2. Set the loss function, optimizer and metrics to print
model.compile(
    loss=keras.losses.CategoricalCrossentropy(),     # <- choose a suitable loss function
    optimizer=keras.optimizers.Adam(learning_rate=0.00001, beta_1=0.9, beta_2=0.999),      # <- you may modify this if you like
    metrics=["accuracy"],    # <- choose a suitable metric, https://www.tensorflow.org/api_docs/python/tf/keras/metrics
)

# train the network
num_epochs = 500      # <- change this value as needed
model.fit(train_dataset, epochs=num_epochs, validation_data=test_dataset, callbacks=[tensorboard_callback, early_stopping, model_checkpoint])

model.save('classifier_model_3')

# predict
#Load the model
model = tf.keras.models.load_model('./classifier_model', custom_objects={'CustomRegularizer': network.CustomRegularizer})
#model.summary()

#Predict on test datasets
prediction=model.predict(test_pc)

#Calculate Accuracy
m = tf.keras.metrics.CategoricalAccuracy()
m.update_state(test_labels, prediction)
Accuracy = m.result().numpy

#Load the normalized test point cloud if necessary
test_pc = pickle.load(open("./dataset/testpc_normalized.pkl", "rb"))
test_labels = pickle.load(open("./dataset/testlabels.pkl", "rb"))
class_ids = pickle.load(open("./dataset/class_ids.pkl", "rb"))

#one-hot --> int
predict_id = np.argmax(prediction, axis=1)
true_id = np.argmax(test_labels, axis=1)

# 3. Display some clouds using matplotlib scatter plot along with true and predicted labels
test_num = len(test_labels)
idx = np.random.randint(test_num)
point_cloud = test_pc[idx,:,:]
p=predict_id[idx]
t=true_id[idx]
utils.visualize_cloud(point_cloud, true_label=class_ids[t], predicted_label=class_ids[p])

# 4. Display a confusion matrix
confusion_matrix=utils.Confusion_Matrix(prediction, test_labels, class_ids)
