import os
import glob
from numpy.core.fromnumeric import shape
import trimesh
import trimesh.sample
import numpy as np
import matplotlib.pyplot as plt
from trimesh.triangles import bounds_tree
# import tensorflow as tf

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

    train_pc = []   # array of training point clouds
    test_pc = []    # array of test point clouds

    train_labels = []   # array of corresponding training labels
    test_labels = []    # array of corresponding test labels

    class_ids = {}   # list of class names

    # get all the folders except the readme file
    folders = glob.glob(os.path.join(data_dir, "[!README]*"))

    for class_id, folder in enumerate(folders):
        print("processing class: {}".format(os.path.basename(folder)))

        # TODO: Fill this part, get the name of the folder (class) and save it
        class_ids[str(class_id)]= folder[11:]

        # get the files in the train folder
        train_files = glob.glob(os.path.join(folder, "train/*"))
        for f in train_files[:10]:
            # TODO: Fill this part
            cad_mesh = trimesh.load(f)  # <- Set path to a .off file
            points = trimesh.sample.sample_surface(cad_mesh, num_points_per_cloud)[0]
            train_pc.append(points)
            train_labels.append(folder[11:])
        # get the files in the test folder
        test_files = glob.glob(os.path.join(folder, "test/*"))
        for f in test_files[:10]:
            # TODO: FIll this part
            cad_mesh = trimesh.load(f)  # <- Set path to a .off file
            points = trimesh.sample.sample_surface(cad_mesh, num_points_per_cloud)[0]
            test_pc.append(points)
            test_labels.append(folder[11:])

    return (np.array(train_pc), np.array(test_pc),
            np.array(train_labels), np.array(test_labels), class_ids)

def semantic_seg_dataset(train_pc, test_pc, train_labels, test_labels, class_ids,num_objects,num_test_data,num_train_data,resampling_size):
    train_pc_seg = []
    test_pc_seg = []
    train_labels_seg = np.zeros((len(class_ids),len(class_ids)))
    test_labels_seg = np.zeros((len(test_pc),len(class_ids)))
    class_ids_seg = np.zeros((len(class_ids),len(class_ids)))
    temp_class_ids = {}
    scene_label = np.zeros((len(class_ids)**num_objects,len(class_ids)))
    for i in range(len(class_ids)):
        temp_class_ids[class_ids[str(i)]] = class_ids_seg[i]

    for c_id in range(len(class_ids)):
        class_ids_seg[c_id,c_id] = 1
    
    for data in range(num_test_data): 
        index = np.random.randint(0,len(test_pc),num_objects)   
        # new = np.random.choice(test_pc[index[0]],size=len(test_pc[index[0]])/num_objects,replace=False)
        new = np.random.choice(test_pc[index[0]].shape[0],size=int(resampling_size/num_objects),replace=False)
        new = test_pc[index[0],new]
        for i in index[1:]:
            axs = np.random.randint(0,6)
            origin = 0
            if axs == 0:
                origin = max(test_pc[i,:,2])
            elif axs == 1:
                origin = min(test_pc[i,:,2])
            elif axs == 2:
                origin = max(test_pc[i,:,1])
            elif axs == 3:
                origin = min(test_pc[i,:,1])
            elif axs == 4:
                origin = max(test_pc[i,:,0])
            elif axs == 5:
                origin = min(test_pc[i,:,0])

            new[:,axs%3] +=  origin
            new = np.concatenate((new,test_pc[i,np.random.choice(test_pc[i].shape[0],size=int(resampling_size/num_objects),replace=False)]),axis=0)

            test_labels_seg[data] += temp_class_ids[test_labels[i]]
        test_pc_seg.append(new)

    for data in range(num_train_data): 
        index = np.random.randint(0,len(train_pc),num_objects)   
        # new = np.random.choice(train_pc[index[0]],size=len(train_pc[index[0]])/num_objects,replace=False)
        new = np.random.choice(train_pc[index[0]].shape[0],size=int(resampling_size/num_objects),replace=False)
        new = train_pc[index[0],new]
        for i in index[1:]:
            axs = np.random.randint(0,6)
            origin = 0
            if axs == 0:
                origin = max(train_pc[i,:,2])
            elif axs == 1:
                origin = min(train_pc[i,:,2])
            elif axs == 2:
                origin = max(train_pc[i,:,1])
            elif axs == 3:
                origin = min(train_pc[i,:,1])
            elif axs == 4:
                origin = max(train_pc[i,:,0])
            elif axs == 5:
                origin = min(train_pc[i,:,0])
            new[:,axs%3] +=  origin
            new = np.concatenate((new,train_pc[i,np.random.choice(train_pc[i].shape[0],size=int(resampling_size/num_objects),replace=False)]),axis=0)
            train_labels_seg[data] += temp_class_ids[train_labels[i]]
        train_pc_seg.append(new)

    k = 0
    for i in class_ids_seg:
        for j in class_ids_seg:
            scene_label[k] = i + j
            k += 1


    return (np.array(train_pc_seg), np.array(test_pc_seg),
            np.array(train_labels_seg), np.array(test_labels_seg), class_ids_seg,np.array(scene_label))

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
    dev_in_metres = 0.002   # <- change this value to change amount of noise
    # add noise to the points
    point_cloud += tf.random.uniform(point_cloud.shape, -dev_in_metres, dev_in_metres, dtype=tf.float64)
    # shuffle points
    point_cloud = tf.random.shuffle(point_cloud)
    return point_cloud, label

if __name__=='__main__':
    a,b,c,d,e = create_point_cloud_dataset('ModelNet10/')
    print([len(a[0]),len(b[0])])
    resampling = int(len(a[0])/3)
    if resampling%2!=0:
        resampling += 1
    train,test,train_l,test_l,class_id,scene_l = semantic_seg_dataset(a, b, c, d, e,2,2,2,resampling)
    visualize_cloud(train[0])
    # print(a)