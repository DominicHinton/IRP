import pandas as pd
import numpy as np
import random
from sklearn.neighbors import NearestNeighbors


def get_euclidean_distance(matrix, point_1, point_2):
  v1 = matrix[point_1]
  v2 = matrix[point_2]
  return np.sqrt(((v1-v2)**2).sum())

def get_manhattan_distance(matrix, point_1, point_2):
  v1 = matrix[point_1]
  v2 = matrix[point_2]
  return (abs(v1 - v2)).sum()

def get_chebyshev_distance(matrix, point_1, point_2):
  v1 = matrix[point_1]
  v2 = matrix[point_2]
  return (abs(v1 - v2)).max()

class SMOTE:
  def __init__(self, original_samples, N, k, random_seed=1):
    random.seed(random_seed)
    np.random.seed(random_seed)
    self.samples = original_samples
    np.random.shuffle(self.samples) # shuffle
    self.N = N // 100 # 100% converts to 1 here
    self.k = k # number for k-nearest-neighbours
    self.T = np.shape(original_samples)[0] # number of minority class samples
    self.n_attributes = np.shape(original_samples)[1]
    self.synthetic = np.zeros((self.T * self.N, self.n_attributes))
    self.new_index = 0
    self.neighbors = NearestNeighbors(n_neighbors=k,
                                      metric="euclidean").fit(self.samples)

  def oversample(self):
    for i in range(self.T):
      sample = self.samples[i]
      nn_array = self.neighbors.kneighbors(sample.reshape(1,-1),
                                           return_distance=False)[0]
      for j in range(self.N):
        chosen_index = random.randint(0, self.k - 1)
        delta = self.samples[nn_array[chosen_index]] - sample
        distance = random.random()
        synthetic_sample = sample + delta * distance
        self.synthetic[self.new_index] = synthetic_sample
        self.new_index += 1
    return self.synthetic



class Outlier_SMOTE:
  def __init__(self, original_samples, N, k, metric="euclidean", random_seed=1):
    random.seed(random_seed)
    np.random.seed(random_seed)
    self.samples = original_samples
    np.random.shuffle(self.samples) # shuffle
    self.N = N // 100 # 100% converts to 1 here
    self.k = k # number for k-nearest-neighbours
    self.T = np.shape(original_samples)[0] # number of minority class samples
    self.n_attributes = np.shape(original_samples)[1]
    self.synthetic = np.zeros((self.T * self.N, self.n_attributes))
    self.new_index = 0
    self.metric = metric
    self.neighbors = NearestNeighbors(n_neighbors=k,
                                      metric=self.metric).fit(self.samples)

    if metric == "euclidean":
      self.get_distance = get_euclidean_distance
    elif metric == "manhattan":
      self.get_distance = get_manhattan_distance
    elif metric == "chebyshev":
      self.get_distance = get_chebyshev_distance

  def oversample(self):
    distance_matrix = np.zeros((self.T, self.T))
    for i in range(self.T):
      for j in range(self.T):
        if j > i:
          distance = self.get_distance(self.samples, i, j)
          distance_matrix[i][j] = distance
          distance_matrix[j][i] = distance

    column_wise_sum_matrix = distance_matrix.sum(axis=0)

    normalised_matrix = column_wise_sum_matrix / sum(column_wise_sum_matrix)

    oversampling_matrix = np.round(self.N * self.T * normalised_matrix).astype(int)

    samples_required = oversampling_matrix.sum()
    # set size of oversampling matrix so that it contains correct number of rows
    self.synthetic = np.zeros((samples_required, self.n_attributes))

    oversampling_list = list(oversampling_matrix)

    for index, element in enumerate(oversampling_list):
      if element == 0: # when no oversampling of this element required
        continue
      sample = self.samples[index]
      nn_array = self.neighbors.kneighbors(sample.reshape(1,-1),
                                           return_distance=False)[0]
      for i in range(element):
        chosen_index = random.randint(0, self.k - 1)
        delta = self.samples[nn_array[chosen_index]] - sample
        distance = random.random()
        synthetic_sample = sample + delta * distance
        self.synthetic[self.new_index] = synthetic_sample
        self.new_index += 1

    return self.synthetic
