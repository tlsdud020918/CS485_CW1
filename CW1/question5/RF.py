import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # set absolute path for importing split_data

from dataset.data import split_data
import numpy as np
import cv2  # OpenCV
from question5.randomforest import *
from randomforest import weakLearner, AxisAligned
# https://github.com/kevin-keraudren/randomforest-python.git






if __name__ == "__main__":
    train_data, train_label, test_data, test_label = split_data(data_path="../dataset/face.mat") # D * N

    params = {'max_depth': 10,
              'min_sample_count': 5,
              'test_count': 30,
              'test_class': AxisAligned()}

    forest = ClassificationForest(10, params) # (n_trees, params)
    forest.fit(train_data.T, train_label)

    test_pred = []
    for data in test_data.T:
        prediction = forest.predict(data)
        test_pred.append(prediction)
    test_pred = np.array(test_pred)

    accuracy = np.mean(test_pred == test_label)

    print("Accuracy: ", accuracy)
    

