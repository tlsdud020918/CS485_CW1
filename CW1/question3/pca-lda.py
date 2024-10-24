import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # set absolute path for importing split_data

from dataset.data import split_data
from question1.eigen import pca, knn_classifier
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import random
from scipy.stats import mode
import matplotlib.pyplot as plt

def pca_projection(base, mean, data, M=50, M0=30, rs=False):
    eig_val, eig_vec = pca(base, mean)
    if rs:
        M1 = M - M0
        random_basis = np.array([np.random.choice(row, size=M1, replace=False) for row in eig_vec[:, M0:]])
        basis = np.hstack((eig_vec[:, :M0], random_basis))
    else:
        basis = eig_vec[:, :M]

    centered_data = data - mean
    projected_data = centered_data.T @ basis

    if rs:
        return projected_data.T, basis, mean
    else:
        return projected_data.T # D * N

def lda(data, label):
    mean_data = np.mean(data, axis=1).reshape(-1, 1)

    Sw = np.zeros((data.shape[0], data.shape[0]))
    Sb = np.zeros((data.shape[0], data.shape[0]))

    for c in range(1, max(label)+1):
        data_c = data[:, label == c]
        n_c = data_c.shape[1]
        mean_c = np.mean(data_c, axis=1).reshape(-1, 1)

        Sw += (data_c - mean_c) @ (data_c - mean_c).T
        Sb += (mean_c - mean_data) @ (mean_c - mean_data).T
    
    val, vec = np.linalg.eig(np.linalg.inv(Sw)@Sb)
    val = val.real
    vec = vec.real

    idx = val.argsort()[::-1]
    val = val[idx]
    vec = vec[:, idx]

    return val, vec

    # U, S, Vt = np.linalg.svd(np.linalg.inv(Sw)@Sb)
    # return U

def lda_projection(base, base_label, data, M=30):
    eig_val, eig_vec = lda(base.astype(np.float64), base_label.astype(np.uint8))
    projected_data = data.T @ eig_vec[:, :M]

    return projected_data # N * D where D = M

def knn_classifier(train_data, train_label, test_data, k=5):
    test_pred = []
    for test in test_data:
        d = np.sqrt(np.sum((train_data - test.reshape(1, -1))**2, axis=1))
        knn_idx = np.argsort(d)[:k]
        knn_labels = train_label[knn_idx]

        unique, counts = np.unique(knn_labels, return_counts=True)
        major = unique[np.argmax(counts)]
        test_pred.append(major)
    
    test_pred = np.array(test_pred)
    
    return test_pred


def pca_lda_classifier(train_data, train_label, mean, test_data, Mpca=335, M0=300, Mlda=51, knn=5, rs=False):
    if rs:
        train_pca_projected, basis, mean = pca_projection(train_data, mean, train_data, Mpca, M0, rs)
        train_lda_projected = lda_projection(train_pca_projected, train_label, train_pca_projected, Mlda)
        test_pca_projected = ((test_data - mean).T @ basis).T
        test_lda_projected = lda_projection(train_pca_projected, train_label, test_pca_projected, Mlda)
    else:
        train_pca_projected = pca_projection(train_data, mean, train_data, M=Mpca)
        train_lda_projected = lda_projection(train_pca_projected, train_label, train_pca_projected, Mlda)
        test_pca_projected = pca_projection(train_data, mean, test_data, M=Mpca)
        test_lda_projected = lda_projection(train_pca_projected, train_label, test_pca_projected, Mlda)


    test_pred = knn_classifier(train_lda_projected, train_label, test_lda_projected)

    # Alternative: utilize KNN library
    #model = KNeighborsClassifier(n_neighbors=knn)
    #model.fit(train_lda_projected, train_label)
    #test_pred = model.predict(test_lda_projected)
    
    return test_pred

def training_data_rs(data, T, subset_rate = 0.75):
    num_total = data.shape[1]
    step = num_total//52
    
    sample = []
    for i in range(T):
        sample_idx = []
        for j in range(0, num_total, step):
            idx = random.sample(range(j, j+step), int(step*subset_rate))
            sample_idx+=idx

        sample_idx = sorted(sample_idx)
        sample_data = data[:, sample_idx]
        sample_label = train_label[sample_idx]
        sample.append([sample_data, sample_label])
    
    return sample


if __name__ == "__main__":
    train_data, train_label, test_data, test_label = split_data(data_path="../dataset/face.mat") # D * N
    mean_face = np.mean(train_data, axis=1).reshape(-1, 1)

    # project to N-1 dim
    #N = train_data.shape[1]
    #projected_train = pca_projection(train_data, mean_face, train_data, M=N-1)
    #projected_test = pca_projection(train_data, mean_face, test_data, M=N-1)
    #projected_mean = np.mean(projected_train, axis=1).reshape(-1, 1)

    # ### pca-lda classifier
    test_pred = pca_lda_classifier(train_data, train_label, mean_face, test_data, Mpca=150, Mlda=50, knn=5)
    accuracy = np.mean(test_pred == test_label)
    print(accuracy)

    # ### pca-lda ensemble
    ensemble_test_pred = []

    train_data_rs = training_data_rs(train_data, 8, subset_rate = 0.5) # training data random sampling, 8 times
    for dataset in train_data_rs:
        sample_mean = np.mean(dataset[0], axis=1).reshape(-1, 1)
        ensemble_test_pred.append(pca_lda_classifier(dataset[0], dataset[1], sample_mean, test_data, Mpca=150, Mlda=50))

    for i in range(8): # feature space random sampling, 8 times
        ensemble_test_pred.append(pca_lda_classifier(train_data, train_label, mean_face, test_data, Mpca=150, M0=145, Mlda=50, rs=True))

    ensemble_test_pred = mode(ensemble_test_pred, axis=0, keepdims=True).mode.flatten()
    ensemble_accuracy = np.mean(ensemble_test_pred == test_label)
    print(ensemble_accuracy)



