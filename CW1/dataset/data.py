import numpy as np
import matplotlib.pylab as plt
import scipy.io as io
import random
import os


random.seed(42)

def split_data(data_path="./face.mat", train_rate=0.8):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    full_data_path = os.path.join(base_dir, data_path)
    face = io.loadmat(full_data_path)

    X = np.array(face['X'])
    L = np.array(face['l'])

    num_total = np.shape(X)[1]
    train_idx = []

    for i in range(0, num_total, 10):
        sample = random.sample(range(i, i+10), int(10*train_rate))
        train_idx+=sample

    train_idx = sorted(train_idx)
    test_idx = sorted(list(set(list(range(0, num_total))) - set(train_idx)))

    train_data = X[:, train_idx]
    train_label = L[:, train_idx].flatten()

    test_data = X[:, test_idx]
    test_label = L[:, test_idx].flatten()

    return train_data, train_label, test_data, test_label

if __name__ == "__main__":
    train_data, train_label, test_data, test_label = split_data()
    # print(test_label)
    io.savemat('face_split.mat', {'train_X': train_data, 'train_L': train_label, 'test_X': test_data, 'test_L': test_label})