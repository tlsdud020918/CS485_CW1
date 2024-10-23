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


