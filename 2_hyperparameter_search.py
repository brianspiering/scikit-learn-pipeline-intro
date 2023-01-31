"""
Perform in-depth hyperparameter search.

"""

import pickle

import numpy as np
import pandas as pd
from   sklearn.base            import BaseEstimator
from   sklearn.decomposition   import PCA
from   sklearn.ensemble        import RandomForestClassifier
from   sklearn.metrics         import f1_score 
from   sklearn.model_selection import RandomizedSearchCV
from   sklearn.pipeline        import Pipeline
from   sklearn.preprocessing   import StandardScaler


class DummyEstimator(BaseEstimator):
    "Pass through class, methods are present but do nothing."
    def fit(self): pass
    def score(self): pass


pipe = Pipeline([('scl', StandardScaler()),
                 ('pca', PCA()),
                 ('clf', DummyEstimator())]) 


# Create space of candidate learning algorithms and their hyperparameters
search_space = [
                {'pca__n_components': range(1, 10),
                 'clf': [RandomForestClassifier(n_jobs=-1)],
                 'clf__criterion': ['gini', 'entropy'],
                 'clf__n_estimators': [50, 100, 150, 200, 250, 300],
                 'clf__max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10],
                 'clf__max_features': ['auto', 'sqrt', 'log2'],
                 'clf__class_weight': [None, 'balanced'],
                 'clf__criterion': ['gini', 'entropy']}
               ]

clf_algos_rand = RandomizedSearchCV(estimator=pipe, 
                                    param_distributions=search_space, 
                                    scoring='f1_weighted',
                                    n_iter=150,
                                    cv=5, 
                                    n_jobs=-1,
                                    verbose=1)

 # Load data
path = "./data/"
X = pd.read_csv(path+"x_train.csv", header=0) # All numeric
y = pd.read_csv(path+"y_train.csv", header=0) # Multi-classification
y = y.values.ravel()

# Search and save
best_model = clf_algos_rand.fit(X, y);
s = pickle.dumps(best_model)