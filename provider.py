import numpy as np


def rotate_point_cloud(batch_data):
  """ Randomly rotate the point clouds to augument the dataset
      rotation is per shape based along up direction
      ref:http://planning.cs.uiuc.edu/node102.html
      Input:
        BxNx3 array, original batch of point clouds
      Return:
        BxNx3 array, rotated batch of point clouds
  """
  rotated_data = np.zeros(batch_data.shape, dtype=np.float64)
  # rotation_angle_z = np.random.uniform() * 2 * np.pi
  # cosval_z = np.cos(rotation_angle_z)
  # sinval_z = np.sin(rotation_angle_z)
  # rotation_matrix_z = np.array([[cosval_z, -sinval_z, 0],
  #                               [sinval_z, cosval_z, 0],
  #                               [0, 0, 1]])
  # rotation_angle_y = np.random.uniform() * 2 * np.pi
  # cosval_y = np.cos(rotation_angle_y)
  # sinval_y = np.sin(rotation_angle_y)
  # rotation_matrix_y = np.array([[cosval_y, 0, sinval_y],
  #                               [0, 1, 0],
  #                               [-sinval_y, 0, cosval_y]])
  # rotation_angle_x = np.random.uniform() * 2 * np.pi
  # cosval_x = np.cos(rotation_angle_x)
  # sinval_x = np.sin(rotation_angle_x)
  # rotation_matrix_x = np.array([[1, 0, 0],
  #                               [0, cosval_x, -sinval_x],
  #                               [0, sinval_x, cosval_x]])
  # rotation_matrix = np.matmul(np.matmul(rotation_matrix_z, rotation_matrix_y), rotation_matrix_x)
  for k in range(batch_data.shape[0]):
    rotation_matrix = get_random_rotation_matrix()
    shape_pc = batch_data[k, ...]
    rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
  return rotated_data

def get_random_rotation_matrix():
  if np.random.random() < 0.5:
    rotation_angle_z = np.random.uniform() * 2 * np.pi
    cosval_z = np.cos(rotation_angle_z)
    sinval_z = np.sin(rotation_angle_z)
    rotation_matrix_z = np.array([[cosval_z, -sinval_z, 0],
                                  [sinval_z, cosval_z, 0],
                                  [0, 0, 1]])
    rotation_angle_y = np.random.uniform() * 2 * np.pi
    cosval_y = np.cos(rotation_angle_y)
    sinval_y = np.sin(rotation_angle_y)
    rotation_matrix_y = np.array([[cosval_y, 0, sinval_y],
                                  [0, 1, 0],
                                  [-sinval_y, 0, cosval_y]])
    rotation_angle_x = np.random.uniform() * 2 * np.pi
    cosval_x = np.cos(rotation_angle_x)
    sinval_x = np.sin(rotation_angle_x)
    rotation_matrix_x = np.array([[1, 0, 0],
                                  [0, cosval_x, -sinval_x],
                                  [0, sinval_x, cosval_x]])
    rotation_matrix = np.matmul(np.matmul(rotation_matrix_z, rotation_matrix_y), rotation_matrix_x)

  else:
    rotation_angle_z = 0
    cosval_z = np.cos(rotation_angle_z)
    sinval_z = np.sin(rotation_angle_z)
    rotation_matrix_z = np.array([[cosval_z, -sinval_z, 0],
                                  [sinval_z, cosval_z, 0],
                                  [0, 0, 1]])
    rotation_angle_y = 0
    cosval_y = np.cos(rotation_angle_y)
    sinval_y = np.sin(rotation_angle_y)
    rotation_matrix_y = np.array([[cosval_y, 0, sinval_y],
                                  [0, 1, 0],
                                  [-sinval_y, 0, cosval_y]])
    rotation_angle_x = 0
    cosval_x = np.cos(rotation_angle_x)
    sinval_x = np.sin(rotation_angle_x)
    rotation_matrix_x = np.array([[1, 0, 0],
                                  [0, cosval_x, -sinval_x],
                                  [0, sinval_x, cosval_x]])
    rotation_matrix = np.matmul(np.matmul(rotation_matrix_z, rotation_matrix_y), rotation_matrix_x)
  return rotation_matrix

def get_avg(points, axis=0):
  sum = 0
  for point in points:
    sum += point[axis]
  return sum/len(points)

def normalize_pc(points):
  normalised = []
  for point in points:
    avg_x = get_avg(point, 0)
    avg_y = get_avg(point, 1)
    avg_z = get_avg(point, 2)

    normalised_points = []

    K = 0
    for p in point:
      K_ = np.sqrt((p[0]-avg_x)**2 + (p[1]-avg_y)**2 + (p[2]-avg_z)**2)
      if K_ > K:
        K = K_

    for p in point:
      x_norm = (p[0]-avg_x)/K
      y_norm = (p[1]-avg_y)/K
      z_norm = (p[2]-avg_z)/K
      normalised_points.append([x_norm, y_norm, z_norm])

    normalised.append(np.array(normalised_points))
  return np.array(normalised)


def normalize_pc_color(points):
  normalised = []
  for point in points:
    avg_x = get_avg(point, 0)
    avg_y = get_avg(point, 1)
    avg_z = get_avg(point, 2)

    normalised_points = []

    K = 0
    for p in point:
      K_ = np.sqrt((p[0]-avg_x)**2 + (p[1]-avg_y)**2 + (p[2]-avg_z)**2)
      if K_ > K:
        K = K_

    for p in point:
      x_norm = (p[0]-avg_x)/(K+1e4)
      y_norm = (p[1]-avg_y)/(K+1e4)
      z_norm = (p[2]-avg_z)/(K+1e4)
      r = p[3]/255
      g = p[4]/255
      b = p[5]/255
      normalised_points.append([x_norm, y_norm, z_norm, r, g, b])

    normalised.append(np.array(normalised_points))
  return np.array(normalised)




def normalize_pc_segmentation(points):
  normalised = []
  for point in points:
    avg_x = get_avg(point, 0)
    avg_y = get_avg(point, 1)
    avg_z = get_avg(point, 2)

    normalised_points = []

    for p in point:
      x_norm = p[0]-avg_x
      y_norm = p[1]-avg_y
      z_norm = p[2]-avg_z
      normalised_points.append([x_norm, y_norm, z_norm])

    normalised.append(np.array(normalised_points))
  return np.array(normalised)
