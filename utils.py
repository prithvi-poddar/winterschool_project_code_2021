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

    for data in range(num_test_data): 
        index = np.random.randint(0, len(test_pc), num_objects)   
        new = test_pc[index[0]]
        label = np.reshape(np.tile(test_labels[index[0]], len(new)), (-1,10))
        # each point gets a column indicating which class it belongs to
        new = np.concatenate((new, label), axis=1)
        for i in index[1:]:
            axs = np.random.randint(0,4)
            origin = 0
            if axs == 0:
                origin = max(test_pc[i,:,0])
            elif axs == 1:
                origin = max(test_pc[i,:,1])
            elif axs == 2:
                origin = min(test_pc[i,:,0])
            elif axs == 3:
                origin = min(test_pc[i,:,1])

            new[:,axs%2] +=  ((-1)**(axs%2))*origin

            label = np.reshape(np.tile(test_labels[i], len(test_pc[i])), (-1,10))
            test_pclabelled = np.concatenate((test_pc[i], label), axis=1)
            new = np.concatenate((new, test_pclabelled), axis=0)
            
        test_pc_seg.append(new)

    for data in range(num_train_data): 
        index = np.random.randint(0, len(train_pc), num_objects)   
        new = train_pc[index[0]]
        label = np.reshape(np.tile(train_labels[index[0]], len(new)), (-1,10))
        # each point gets a column indicating which class it belongs to
        new = np.concatenate((new, label), axis=1)
        for i in index[1:]:
            axs = np.random.randint(0,4)
            origin = 0
            if axs == 0:
                origin = max(train_pc[i,:,0])
            elif axs == 1:
                origin = max(train_pc[i,:,1])
            elif axs == 2:
                origin = min(train_pc[i,:,0])
            elif axs == 3:
                origin = min(train_pc[i,:,1])

            new[:,axs%2] +=  ((-1)**(axs%2))*origin

            label = np.reshape(np.tile(train_labels[i], len(train_pc[i])), (-1,10))
            train_pclabelled = np.concatenate((train_pc[i], label), axis=1)
            new = np.concatenate((new, train_pclabelled), axis=0)
            
        train_pc_seg.append(new)

    # pc_seg structure is scene x points x (x,y,z, hot encoded class)
    return (np.array(train_pc_seg[:,:,:3]), np.array(test_pc_seg[:,:,:3]), np.array(train_pc_seg[:,:,-9:]), np.array(test_pc_seg[:,:,-9:]))

def visualize_cloud(point_cloud, true_label='', predicted_label=''):
    """
    Utility function to visualize a point cloud
    :param point_cloud: input point cloud
    :type point_cloud: numpy array
    """
    if true_label=='':
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2])
        plt.show()
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2])
        ax.set_xlim(-1,1)
        ax.set_ylim(-1,1)
        ax.set_zlim(-1,1)
        ax.text(x=0, y=0, z=1.2,s="true label: "+true_label, fontsize=10)
        ax.text(x=0, y=0, z=1,s="predicted label: "+predicted_label, fontsize=10)    
        plt.show()
    return


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
    dev_in_metres = 0.05  # <- change this value to change amount of noise
    # add noise to the points
    point_cloud += tf.random.uniform(point_cloud.shape, -dev_in_metres, dev_in_metres, dtype=tf.float64)
    # shuffle points
    # point_cloud = tf.random.shuffle(point_cloud)
    return point_cloud, label

def Confusion_Matrix(prediction, labels):
    """
    Plot the confusion matrix for classfication
    :param prediction: predicted labels
    :type  prediction: tensor
    :param labels    : True labels
    :type label      : tensor
    :param class_ids : id and items
    :type label      : dictionary
    :return: the comfusion matrix
    :rtype:  numpy array
    """
    classes=class_ids.values()
    num_classes = len(classes)
    confusion_matrix = np.zeros([num_classes,num_classes])
    predict_id = np.argmax(prediction, axis=1)
    true_id = np.argmax(test_labels, axis=1)

    for i in range(len(predict_id)):
        p = predict_id[i]
        t = true_id[i]
        confusion_matrix[p,t]+=1
    
    for i in range(num_classes):
        confusion_matrix[i,:]/= np.sum(confusion_matrix[i,:]) 
    
    plt.figure()
    plt.imshow(confusion_matrix,cmap=plt.cm.Oranges)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xlabel('True Labels')
    plt.ylabel('Predicted Labels')
    tick_marks = classes
    plt.tight_layout()
    plt.xticks(np.arange(num_classes),classes,rotation=-60)
    plt.yticks(np.arange(num_classes), classes)
    
    return confusion_matrix
