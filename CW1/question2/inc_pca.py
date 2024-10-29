import numpy as np
import time
import psutil
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset import data
from question1 import eigen

def memory_usage(message: str = 'debug'):
    # current process RAM usage
    p = psutil.Process()
    rss = p.memory_info().rss / 2 ** 20 # Bytes to MB
    print(f"[{message}] memory usage: {rss: 10.5f} MB")
    return rss

def gram_schmidt(mat1, mat2):
    q, _ = np.linalg.qr(np.hstack((mat1, mat2)))
    phi = q[:, :mat1.shape[1] + mat2.shape[1]]
    
    return phi

def batch_preprocessing(batch, batch_mean, n_components):
    eigenval, eigenvec = eigen.pca(batch, batch_mean)
    return eigenval[:n_components], eigenvec[:, :n_components]

class IncrementalPCA:
    def __init__(self, n_components, check_performance = False, step_save = False):
        self.n_components = n_components
        self.n_samples = 0 # mu
        self.mean = None # N3
        self.eigenvec = None # P3
        self.eigenval = None # Λ3
        self.check_performance = check_performance
        self.step_save = step_save
        if (check_performance == True):
            self.new_batch_t = []
            self.new_batch_m = []
            self.combine_t = []
            self.combine_m = []
        if (step_save == True):
            self.step_mean = []
            self.step_eigenvec = []
            self.step_eigenval = []

    def partial_fit(self, batch):
        # preprocessing new batch, set variables
        N1 = self.n_samples
        N2 = batch.shape[1]
        N3 = N1 + N2

        mu1 = self.mean
        mu2 = np.mean(batch, axis=1).reshape(-1, 1)

        start = time.time()
        V2, P2 = batch_preprocessing(batch, mu2, self.n_components)
        end = time.time()
        # ---------------------------- first batch exception ----------------------------
        if self.mean is None:
            self.n_samples = N2
            self.mean = mu2
            self.eigenvec = P2
            self.eigenval = V2

            if self.check_performance:
                t = end - start
                print(f"New batch eigen decomposition time: {t:.5f} sec")
                mem = memory_usage(message='incremental_pca_eigen_decomposition')
                self.new_batch_t.append(t)
                self.new_batch_m.append(mem)
                self.combine_t.append(0.0)
                self.combine_m.append(0.0)
            if self.step_save:
                self.step_mean.append(self.mean)
                self.step_eigenvec.append(self.eigenvec)
                self.step_eigenval.append(self.eigenval)
            return
        #--------------------------- first batch exception end ----------------------------

        V1 = self.eigenval
        P1 = self.eigenvec

        S1 = P1 @ np.diag(V1) @ P1.T
        S2 = P2 @ np.diag(V2) @ P2.T

        if self.check_performance:
            t = end - start
            print(f"New batch eigen decomposition time: {t:.5f} sec")
            mem = memory_usage(message='incremental_pca_eigen_decomposition')
            self.new_batch_t.append(t)
            self.new_batch_m.append(mem)

        # ---------------------------- start combining ----------------------------
        # 1. combined mean mu3
        mu3 = (N1 * mu1 + N2 * mu2) / (N1+N2)

        # 2. combined covariance matrix S3
        temp = mu1 - mu2
        S3 = (N1 / N3 * S1) + (N2 / N3 * S2) + (N1*N2 / (N3*N3) * (temp @ temp.T))

        # 3. orthogonalization
        phi = gram_schmidt(gram_schmidt(P1, P2), temp)

        # 4. calculate new eigenval V3, eigenvec P3
        start = time.time()
        val, vec = np.linalg.eig(phi.T @ S3 @ phi)
        idx = val.argsort()[::-1]
        val = val[idx]
        vec = vec[:, idx]

        V3 = val[:self.n_components]
        P3 = phi @ vec[:,:self.n_components]

        end = time.time()

        if self.check_performance:
            t = end - start
            print(f"combined covariance matrix decomposition time: {t:.5f} sec")
            mem = memory_usage(message='incremental_pca_eigen_decomposition')
            self.combine_t.append(t)
            self.combine_m.append(mem)
        
        # update self varialbles 
        self.n_samples = N3
        self.mean = mu3
        self.eigenvec = P3
        self.eigenval = V3

        if self.step_save:
            self.step_mean.append(self.mean)
            self.step_eigenvec.append(self.eigenvec)
            self.step_eigenval.append(self.eigenval)

        print("result shape: ", P3.shape, V3.shape)

    def fit(self, A, batch_size):
        # Split data into batches
        n_samples = A.shape[1]
        for i in range(0, n_samples, batch_size):
            batch = A[:, i:i + batch_size]
            self.partial_fit(batch)

        # Perform eigenvalue decomposition after processing all batches
        return self.eigenval, self.eigenvec

    def transform(self, A):
        # dimensionality reduction: image space -> PCA space
        A_centered = A - self.mean
        return self.eigenvec.T @ A_centered

    def inverse_transform(self, A_reduced):
        # reconstruction: PCA space -> image space
        print(A_reduced.shape, self.eigenvec.shape)
        return self.mean + (self.eigenvec @ A_reduced)

def imaging (data, reconstructed_data):
    # show only the first image of test dataset
    recon_fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.imshow(data[:, 0].reshape((46, -1)).T, cmap='gray')
    ax2.imshow(reconstructed_data[:, 0].reshape((46, -1)).T, cmap='gray')
    ax1.set_title('Original')
    ax2.set_title('Reconstructed with bases')

    plt.show()

# Example usage
if __name__ == "__main__":
    # Generate synthetic data (e.g., 2576 dimensions, 416 samples)

    # Initialize Incremental PCA
    ipca = IncrementalPCA(n_components=50)

    data_path = "../dataset/face.mat"
    train_data, train_label, test_data, test_label = data.split_data(data_path)

    # 1. additional seperation of the training data
    train_data1 = train_data[:, 0::4]
    train_data2 = train_data[:, 1::4]
    train_data3 = train_data[:, 2::4]
    train_data4 = train_data[:, 3::4]

    train_label1 = train_label[0::4]
    train_label2 = train_label[1::4]
    train_label3 = train_label[2::4]
    train_label4 = train_label[3::4]

    train_data_reconstruct = np.hstack([train_data1, train_data2, train_data3, train_data4])
    train_label_reconstruct = np.concatenate([train_label1, train_label2, train_label3, train_label4])

    # Fit Incremental PCA to the dataset
    eigvals, eigvecs = ipca.fit(train_data_reconstruct, 104)

    print("Eigenvalues (Top 50):", np.diag(eigvals))
    print("Shape of Eigenvectors:", eigvecs.shape)

    # KNN test
    train_PCA_result = ipca.transform(train_data_reconstruct)
    test_PCA_result = ipca.transform(test_data)

    imaging (train_data_reconstruct, ipca.inverse_transform(train_PCA_result))

    model = KNeighborsClassifier(n_neighbors=5, weights='distance')
    model.fit(train_PCA_result.T, train_label_reconstruct)
    result = model.score(test_PCA_result.T, test_label)
    print(result)
