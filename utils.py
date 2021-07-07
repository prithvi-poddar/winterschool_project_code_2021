import enum
import os
import glob
import trimesh
import trimesh.sample
import numpy as np
import matplotlib.pyplot as plt
from trimesh.triangles import bounds_tree
import tensorflow as tf


def create_point_cloud_dataset(data_dir, num_points_per_cloud=1024):
    """
    Given the path to the ModelNet10 dataset, samples the models and creates point clouds
    :param data_dir: path to the ModelNet10 dataset
    :type data_dir: str
    :param num_points_per_cloud: number of points to sample per cloud. 1024, 2048....
    :type num_points_per_cloud: int
    :return: tuple of numpy array containing training and test point clouds, their corresponding labels and a list of
    class IDs
    :rtype: tuple
    """

    train_pc = []  # array of training point clouds
    test_pc = []  # array of test point clouds

    train_labels = []  # array of corresponding training labels
    test_labels = []  # array of corresponding test labels

    class_ids = {}  # list of class names

    # get all the folders except the readme file
    folders = glob.glob(os.path.join(data_dir, "[!README]*"))

    for class_id, folder in enumerate(folders):
        print("processing class: {}".format(os.path.basename(folder)))

        # TODO: Fill this part, get the name of the folder (class) and save it
        class_ids[class_id] = os.path.basename(folder)

        # get the files in the train folder
        train_files = glob.glob(os.path.join(folder, "train/*"))
        for f in train_files:
            # TODO: Fill this part
            points = trimesh.sample.sample_surface(trimesh.load(f), num_points_per_cloud)[0]
            train_pc.append(points)
            train_labels.append(class_id)
        # get the files in the test folder
        test_files = glob.glob(os.path.join(folder, "test/*"))
        for f in test_files:
            # TODO: FIll this part
            points = trimesh.sample.sample_surface(trimesh.load(f), num_points_per_cloud)[0]
            test_pc.append(points)
            test_labels.append(class_id)

    encoded_train_labels = []
    for idx, label in enumerate(train_labels):
        one_hot = np.zeros(10)
        one_hot[label] = 1.
        encoded_train_labels.append(one_hot)
    encoded_train_labels = np.array(encoded_train_labels)

    encoded_test_labels = []
    for idx, label in enumerate(test_labels):
        one_hot = np.zeros(10)
        one_hot[label] = 1.
        encoded_test_labels.append(one_hot)
    encoded_test_labels = np.array(encoded_test_labels)

    return (np.array(train_pc), np.array(test_pc),
            np.array(encoded_train_labels), np.array(encoded_test_labels), class_ids)


def semantic_seg_dataset(data_dir, num_objects, num_test_data, num_train_data, num_points_per_cloud=1024):
    """
    creates a semantic dataset and returns train points, test points, train labels, test labels
    num_objects: number of objects per scene
    num_train_data: number of training objects you want to create
    num_test_data: number of testing objects you want to create
    """
    train_pc, test_pc, train_labels, test_labels, class_ids = create_point_cloud_dataset(data_dir, num_points_per_cloud)
    train_pc_seg = []
    test_pc_seg = []
    train_seg_labels = []
    test_seg_labels = []

    for data in range(num_train_data): 
        index = np.random.randint(0, len(train_pc), num_objects)   
        scene = train_pc[index[0]]
        label = np.reshape(np.tile(train_labels[index[0]], len(scene)), (-1,10))
        for i in index[1:]:
            axs = np.random.randint(0, 2)
            origin = 0
            if axs == 0:
                dim_scene = np.abs(max(scene[:,0])) + np.abs(min(scene[:,0]))
                dim_new = np.abs(max(train_pc[i,:,0])) + np.abs(min(train_pc[i,:,0]))
                origin =  max(dim_scene, dim_new)
            elif axs == 1:
                dim_scene = np.abs(max(scene[:,1])) + np.abs(min(scene[:,1]))
                dim_new = np.abs(max(train_pc[i,:,1]))- np.abs(min(train_pc[i,:,1]))
                origin =  max(dim_scene, dim_new)
            elif axs == 2:
                dim_scene = np.abs(max(scene[:,2])) + np.abs(min(scene[:,2]))
                dim_new = np.abs(max(train_pc[i,:,2]))- np.abs(min(train_pc[i,:,2]))
                origin =  max(dim_scene, dim_new)

            scene[:,axs%3] +=  ((-1)**(np.random.randint(0, 1)))*origin

            label_i = np.reshape(np.tile(train_labels[i], len(train_pc[i])), (-1,10))
            label = np.concatenate((label, label_i), axis=0)
            scene = np.concatenate((scene, train_pc[i]), axis=0)

        train_pc_seg.append(scene)
        train_seg_labels.append(label)

    for data in range(num_test_data): 
        index = np.random.randint(0, len(test_pc), num_objects)   
        scene = test_pc[index[0]]
        label = np.reshape(np.tile(test_labels[index[0]], len(scene)), (-1,10))
        for i in index[1:]:
            axs = np.random.randint(0, 2)
            origin = 0
            if axs == 0:
                dim_scene = np.abs(max(scene[:,0])) + np.abs(min(scene[:,0]))
                dim_new = np.abs(max(test_pc[i,:,0])) + np.abs(min(test_pc[i,:,0]))
                origin =  max(dim_scene, dim_new)
            elif axs == 1:
                dim_scene = np.abs(max(scene[:,1])) + np.abs(min(scene[:,1]))
                dim_new = np.abs(max(test_pc[i,:,1])) + np.abs(min(test_pc[i,:,1]))
                origin =  max(dim_scene, dim_new)
            elif axs == 2:
                dim_scene = np.abs(max(scene[:,2])) + np.abs(min(scene[:,2]))
                dim_new = np.abs(max(test_pc[i,:,2])) + np.abs(min(test_pc[i,:,2]))
                origin =  max(dim_scene, dim_new)
            scene[:,axs%3] +=  ((-1)**(np.random.randint(0, 1)))*origin

            label_i = np.reshape(np.tile(test_labels[i], len(test_pc[i])), (-1,10))
            label = np.concatenate((label, label_i), axis=0)
            scene = np.concatenate((scene, test_pc[i]), axis=0)
            
        test_pc_seg.append(scene)
        test_seg_labels.append(label)

    return (np.array(train_pc_seg), np.array(test_pc_seg), np.array(train_seg_labels), np.array(test_seg_labels))

def visualize_cloud(point_cloud):
    """
    Utility function to visualize a point cloud
    :param point_cloud: input point cloud
    :type point_cloud: numpy array
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2])
    plt.show()


def add_noise_and_shuffle(point_cloud, label):
    """
    Adds noise to a point cloud and shuffles it
    :param point_cloud: input point cloud
    :type point_cloud: tensor
    :param label: corresponding label
    :type label: tensor
    :return: the processed point cloud and the label
    :rtype: tensors
    """
    dev_in_metres = 0.002  # <- change this value to change amount of noise
    # add noise to the points
    point_cloud += tf.random.uniform(point_cloud.shape, -dev_in_metres, dev_in_metres, dtype=tf.float64)
    # shuffle points
    point_cloud = tf.random.shuffle(point_cloud)
    return point_cloud, label
