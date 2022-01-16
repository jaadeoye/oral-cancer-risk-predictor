import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTENC
from imblearn.over_sampling import ADASYN
from imblearn.combine import SMOTEENN
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
import pickle
#import data
df_train = pd.read_csv('/Users/jaadeoye/Desktop/screen_ml/screen_full3.csv')
features = ['V1', 'V3', 'V4', 'V6', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14',
       'V15', 'V16', 'V17', 'V18', 'V20', 'V21', 'V22', 'V23', 'V25', 'V27',
       'V28', 'V29', 'V30', 'V31', 'V33', 'V34', 'V35']
x = df_train[features]
y = df_train.Status
#train model
sm = SMOTEENN(random_state=0)
x_res, y_res = sm.fit_resample(x,y)
logreg = KNeighborsClassifier(n_neighbors = 2)
logreg.fit(x_res,y_res)
#pickle
pickle.dump(logreg, open('knn', 'wb'))
