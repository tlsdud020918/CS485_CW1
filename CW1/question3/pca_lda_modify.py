import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # set absolute path for importing split_data

from dataset.data import split_data
from question1.eigen import pca, knn_classifier
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import KNeighborsClassifier
import random
from scipy.stats import mode
import matplotlib.pyplot as plt

""" 
<<<<<KEY VARIABLES>>>>>
randomisation in feature space: PCA 이후 feature를 뽑는 것에서 random 요소 추가
randomisation on data samples (i.e. bagging)
the number of base models: PCA-LDA-KNN으로 이어지는 모델을 몇개 만드는가 
the randomness parameter: random의 수준 -> 더 랜덤하게 vs 덜 랜덤하게
fusion rules: model을 합치는 방법 -> classification: majority voting이나 posterior distribution avg 등...

<<<<<KEY EXPERIMENTS: 측정해야 하는 metric들>>>>>
the error of the committee machine vs the average error of individual models: 합친 모델 vs 각각의 모델
recognition accuracy and confusion matrix  
""" 

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

def lda_projection(base, base_label, data, M=30):
    # base, base label로 train 이후 data에 대한 output을 내놓는 function
    lda_model = LDA(n_components = M)
    #print(M, base.shape, base_label.shape)
    lda_model.fit(base.T, base_label)  
    projected_data = lda_model.transform(data.T)

    return projected_data # N * D where D = M

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
    
    # Alternative: utilize KNN library
    model = KNeighborsClassifier(n_neighbors=knn)
    model.fit(train_lda_projected, train_label)
    test_pred = model.predict(test_lda_projected)

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

"""
optimization 
1. mpca, n_nearest 먼저 고정하기, Q1 실험에서 결정
2. mlda 고정: ensemble 전 그냥 pca-lda에서 결정
3. bagging subset rate, m0, model_num 고정: ensemble에서만 있는 변수
"""
if __name__ == "__main__":
    train_data, train_label, test_data, test_label = split_data(data_path="../dataset/face.mat") # D * N
    mean_face = np.mean(train_data, axis=1).reshape(-1, 1)

    mpca = 150
    m0 = 145
    mlda = 50
    n_nearest = 5
    model_num = 8

    # pca-lda classifier
    test_pred = pca_lda_classifier(train_data, train_label, mean_face, test_data, Mpca=mpca, Mlda=mlda, knn=n_nearest)
    accuracy = np.mean(test_pred == test_label)
    print(accuracy)
    #print(test_pred)

    # pca-lda ensemble
    ensemble_test_pred = []

    train_data_rs = training_data_rs(train_data, model_num, subset_rate = 0.7) # training data random sampling, 8 times
    for dataset in train_data_rs:
        sample_mean = np.mean(dataset[0], axis=1).reshape(-1, 1)
        ensemble_test_pred.append(pca_lda_classifier(dataset[0], dataset[1], sample_mean, test_data, Mpca=mpca, Mlda=mlda))

    for i in range(model_num): # feature space random sampling, model_num times
        ensemble_test_pred.append(pca_lda_classifier(train_data, train_label, mean_face, test_data, Mpca=mpca, M0=m0, Mlda=mlda, rs=True))

    ensemble_test_pred = mode(ensemble_test_pred, axis=0, keepdims=True).mode.flatten()
    ensemble_accuracy = np.mean(ensemble_test_pred == test_label)
    print(ensemble_accuracy)
    #print(ensemble_test_pred)


