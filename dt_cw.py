import pandas as pd
from sklearn import metrics
import numpy as np
from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek
from sklearn.tree import DecisionTreeClassifier
import pickle
#import data
df_train = pd.read_csv('/Users/jaadeoye/Desktop/screen_ml/screen_full3.csv')
features = ['V1', 'V3', 'V4', 'V6', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14',
       'V15', 'V16', 'V17', 'V18', 'V20', 'V21', 'V22', 'V23', 'V25', 'V27',
       'V28', 'V29', 'V30', 'V31', 'V33', 'V34', 'V35']
x = df_train[features]
y = df_train.Status
#train model
logreg = DecisionTreeClassifier(random_state=0, max_depth=3, class_weight="balanced")
logreg.fit(x,y)
#pickle
pickle.dump(logreg, open('dt2', 'wb'))
