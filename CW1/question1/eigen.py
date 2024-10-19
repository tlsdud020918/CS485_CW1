import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # set absolute path for importing split_data

from dataset.data import split_data
import numpy as np
import matplotlib.pyplot as plt
import time
import psutil


train_data, train_label, test_data, test_label = split_data(data_path="../dataset/face.mat")
mean_face = np.mean(train_data, axis=1).reshape(-1, 1)

def memory_usage(message=''):
    # current process RAM usage
    p = psutil.Process()
    rss = p.memory_info().rss / 2 ** 20 # Bytes to MB
    print(f"[{message}] memory usage: {rss: 10.5f} MB")

def pca(A, mean_A, low=True):
    start = time.time()

    total_num = A.shape[1]
    A = A - mean_A

    if low==True: 
        S = (A.T@A)/total_num
        val, vec = np.linalg.eig(S)
        vec = A@vec
        vec = vec / np.linalg.norm(vec, axis=0)

    else:
        S = (A@A.T)/total_num
        val, vec = np.linalg.eig(S)

    # sort eigenvalues in descent order
    idx = val.argsort()[::-1]
    val = val[idx]
    vec = vec[:, idx]

    end = time.time()
    print(f"{end - start: .5f} sec")
    memory_usage(message='pca')

    return val, vec

def mean_face_reconstruction(mean_face):
    reconstructed_mean_face = mean_face.reshape(46, -1).T
    plt.imshow(reconstructed_mean_face, cmap='gray')
    plt.title('Mean Face')
    plt.show()

def plot_eig_val(val):
    A = val[val > 1]
    num = A.shape[0]
    x = np.arange(0,num)
    y = A.flatten()

    plt.plot(x, y)
    plt.ylabel('eigen value')
    plt.title('Eigen Values')
    plt.show()

def face_reconstruction(data, M=50):
    eig_val, eig_vec = pca(train_data, mean_face)

    centered_data = data - mean_face

    projected_data = centered_data.T@eig_vec[:, :M] # project data onto PCA vector space with M=30
    reconstructed_data = mean_face + eig_vec[:, :M]@projected_data.T

    theoretical_error = np.sum(eig_val[M:])
    reconstruction_error = np.sum((data - reconstructed_data)**2, axis=0).mean()

    # show only the first image of test dataset
    recon_fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.imshow(data[:, 0].reshape((46, -1)).T, cmap='gray')
    ax2.imshow(reconstructed_data[:, 0].reshape((46, -1)).T, cmap='gray')
    ax1.set_title('Original')
    ax2.set_title('Reconstructed with {} bases'.format(M))

    plt.show()

    return theoretical_error, reconstruction_error # same when using train dataset

def knn_classifier(train_data, train_label, test_data, k=5, M=100):
    train_mean = np.mean(train_data, axis=1).reshape(-1, 1)
    eig_val, eig_vec = pca(train_data, train_mean)

    train_centered = train_data - train_mean
    test_centered = test_data - train_mean
    train_projected = train_centered.T@eig_vec[:, :M]
    test_projected = test_centered.T@eig_vec[:, :M]

    start = time.time()

    test_pred = []
    for test in test_projected:
        d = np.sqrt(np.sum((train_projected - test.reshape(1, -1))**2, axis=1))
        knn_idx = np.argsort(d)[:k]
        knn_labels = train_label[knn_idx]

        unique, counts = np.unique(knn_labels, return_counts=True)
        major = unique[np.argmax(counts)]
        test_pred.append(major)
    
    test_pred = np.array(test_pred)

    end = time.time()
    print(f"{end - start: .5f} sec")
    memory_usage(message='knn-classifier')
    
    return test_pred



# # mean face reconstruction
# mean_face_reconstruction(mean_face)

# # visualize PCA eigen values that are larger than 1
# val, vec = pca(train_data, mean_face)
# plot_eig_val(val[val > 1])

# # reconstruct data based on pca using train data
# te, re = face_reconstruction(test_data)
# print(te, re)

# # knn classifier
# test_pred = knn_classifier(train_data, train_label, test_data)
# accuracy = np.mean(test_pred == test_label)
# print(accuracy)
