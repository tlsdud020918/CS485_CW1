from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset import data
from question1 import eigen

def pca_lib (m_pca, n_neighbors):
    pca = PCA(n_components=m_pca, svd_solver='full') # same result as eigen.py  
    knn = KNeighborsClassifier(n_neighbors=n_neighbors) 

    model = Pipeline(steps=[('pca', pca), ('knn', knn)])
    model.fit(train_data.T, train_label)
    result = model.score(test_data.T, test_label)
    print(result)

def pca_lda (m_pca, m_lda, n_neighbors):
    mean_A = np.mean(train_data, axis=1).reshape(-1, 1) 

    # low PCA
    val, vec = eigen.pca(train_data, mean_A, low=True)  
    train_pca = vec[:, :m_pca].T @ (train_data - mean_A)

    # LDA
    lda = LDA(n_components=m_lda)
    lda.fit(train_pca.T, train_label)  
    train_lda = lda.transform(train_pca.T)

    # KNN Classifier
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(train_lda, train_label)  # Fit KNN on PCA-reduced training data
    
    # testing data
    test_reduced_data = vec[:, :m_pca].T @ (test_data - mean_A)  
    test_lda = lda.transform(test_reduced_data.T)
    result = model.score(test_lda, test_label)
    print(result)

    prediction = model.predict(test_lda)
    return prediction

def np_to_img (input):
    img = input.reshape(46, -1).T
    plt.imshow(img, cmap='gray')
    plt.show()

def show_example_result (prediction, test_label, num):
    # 보고서 형식에 따라 수정예정
    return

if __name__ == "__main__":
    data_path = "../dataset/face.mat"
    train_data, train_label, test_data, test_label = data.split_data(data_path)

    m_pca = 150
    m_lda = 50
    n_neighbors = 5

    #pca_lib (m_pca, n_neighbors)

    prediction = pca_lda(m_pca, m_lda, n_neighbors)
    print(prediction)

    # confusion matrix code
    conf_mtx = confusion_matrix(test_label, prediction, labels=np.unique(test_label))
    img1 = ConfusionMatrixDisplay(confusion_matrix=conf_mtx, display_labels=np.unique(test_label))
    img1.plot() 
    plt.show()

    # example failure and success images