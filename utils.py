import enum
import os
import glob
import trimesh
import trimesh.sample
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from trimesh.triangles import bounds_tree
import tensorflow as tf
import pickle

colors =        {'1':	[0,255,0],
                 '2':	[0,0,255],
                 '3':	[0,255,255],
                 '4':   [255,255,0],
                 '5':   [255,0,255],
                 '6':   [100,100,255],
                 '7':   [200,200,100],
                 '8':   [170,120,200],
                 '9':   [255,0,0],
                 '10':  [200,100,100],
                 '11':  [10,200,100],
                 '12':  [200,200,200],
                 '13':  [50,50,50]}  

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

def semantic_data_mod(pc, labels, num_data, num_objects):
    pc_seg, labels_seg = [], []
    for data in range(num_data): 
        index = np.random.randint(0, len(pc), num_objects)
        scene = pc[index[0]]
        color_data = np.reshape(np.tile(colors['13'], len(scene)), (-1,3))
        scene = np.concatenate((scene, color_data), axis=1)
        label = np.reshape(np.tile(labels[index[0]], len(scene)), (-1,10))
        for i in index[1:]:
            # scale to preserve different scene size
            for j in range(0, 3):
                m = max(pc[i,:,j])
                n = min(pc[i,:,j])
                m_sc = max(scene[:,j])
                n_sc = max(scene[:,j])
                if m > m_sc or n < n_sc:
                    pc[i,:,j] = (m_sc - n_sc)*(train_pc[i,:,j] - n)/(m - n)

            # translate 
            axs = np.random.randint(0, 3)
            dim_scene = np.abs(max(scene[:,axs])) + np.abs(min(scene[:,axs]))
            dim_new = np.abs(max(pc[i,:,axs])) + np.abs(min(pc[i,:,axs]))
            origin =  max(dim_scene, dim_new)
            scene[:,axs] +=  ((-1)**(np.random.randint(0, 2)))*origin

            label_i = np.reshape(np.tile(labels[i], len(pc[i])), (-1,10))
            label = np.concatenate((label, label_i), axis=0)
            color_data = np.reshape(np.tile(colors[str(np.random.randint(1,13))], len(pc[i])), (-1,3))
            colored_object = np.concatenate((pc[i], color_data), axis=1)
            scene = np.concatenate((scene, colored_object), axis=0)
        pc_seg.append(scene)
        labels_seg.append(label)
    return pc_seg, labels_seg

def semantic_seg_dataset(data_dir, num_objects, num_test_data, num_train_data, num_points_per_cloud=1024):
    """
    creates a semantic dataset and returns train points, test points, train labels, test labels
    num_objects: number of objects per scene
    num_train_data: number of training objects you want to create
    num_test_data: number of testing objects you want to create
    """
    train_pc, test_pc, train_labels, test_labels, class_ids = create_point_cloud_dataset(data_dir, num_points_per_cloud)
    train_pc_seg, train_seg_labels = semantic_data_mod(train_pc, train_labels, num_train_data, num_objects)
    test_pc_seg, test_seg_labels = semantic_data_mod(test_pc, test_labels, num_test_data, num_objects)

    return (np.array(train_pc_seg), np.array(test_pc_seg), np.array(train_seg_labels), np.array(test_seg_labels), np.array(class_ids))

def visualize_cloud(point_cloud, true_label='', predicted_label=''):
    """
    Utility function to visualize a point cloud
    :param point_cloud: input point cloud
    :type point_cloud: numpy array
    """
    if true_label=='':
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2])
        plt.show()
    else:
        fig = plt.figure()
        ax = Axes3D(fig)
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

def Confusion_Matrix(prediction, labels, class_ids):
    """
    Plot the confusion matrix for classfication
    :param predict_id: predicted labels
    :type  predict_id: tensor
    :param true_id    : True labels
    :type true_id      : tensor
    :param class_ids : id and items
    :type  class_ids      : dictionary
    :return: the comfusion matrix
    :rtype:  numpy array
    """
    classes=class_ids.values()
    num_classes = len(classes)
    confusion_matrix = np.zeros([num_classes,num_classes])

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


if __name__=='__main__':
    train_pc, test_pc, train_labels, test_labels = semantic_seg_dataset('ModelNet10/', 4, 1000, 4000, 1024)
    pickle.dump(train_pc, open("train_seg4.pkl", "wb"))
    pickle.dump(test_pc, open("test_seg4.pkl", "wb"))
    pickle.dump(train_labels, open("train_seg4_labels.pkl", "wb"))
    pickle.dump(test_labels, open("test_seg4_labels.pkl", "wb"))
    train_pc, test_pc, train_labels, test_labels = semantic_seg_dataset('ModelNet10/', 2, 1000, 4000, 2048)
    pickle.dump(train_pc, open("train_seg2.pkl", "wb"))
    pickle.dump(test_pc, open("test_seg2.pkl", "wb"))
    pickle.dump(train_labels, open("train_seg2_labels.pkl", "wb"))
    pickle.dump(test_labels, open("test_seg2_labels.pkl", "wb"))
    