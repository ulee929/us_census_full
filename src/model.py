import math

import numpy as np
import pandas as pd
import sklearn

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, mean_squared_error
from sklearn.preprocessing import StandardScaler


class DataPreprocessing():
    '''Data preprocessing before modeling'''

    def __init__(self, features, target):
        '''
        Parameters
        ----------
        features : array
            feature values as array (i.e., .values from pandas)
        target : array
            feature values as array (i.e., .values from pandas)
        '''
        self.features = features
        self.target = target

    def split_and_standardize_data(self):
        '''Splits data into train and test sets and standardizes features

        Returns
        ----------
        X_train : array
            array containing the sample points for X train
        X_test : array
            array containing the sample points for X test
        y_train : array
            array containing the sample points for Y train
        y_test : array
            array containing the sample points for Y test
        '''
        # Set random seed so each split is the same (for model comparison)
        np.random.seed(42)

        # Use sklearn's train_test_split to split data
        X_train, X_test, y_train, y_test = train_test_split(self.features, self.target, shuffle=True)

        # Instantiate the scaler
        scaler = StandardScaler()
        # Scale X train and test data
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test


class Models():
    '''Modeling data'''

    def __init__(self, model, features, target, pca=False, smote=False):
        '''
        Parameters
        ----------
        model : sklearn base estimator
            specify which model for data
        features : array
            feature values as array (i.e., .values from pandas)
        target : array
            feature values as array (i.e., .values from pandas)
        pca : boolean
            whether or not to apply PCA to data
        smote : boolean
            whether or not to try SMOTE to reduce dimensionality and balance classes
        '''
        self.model = model
        self.features = features
        self.target = target
        self.pca = pca
        self.smote = smote

    def model_data(self):
        X_train_scaled, X_test_scaled, y_train, y_test = DataPreprocessing(self.features, self.target).split_and_standardize_data()

        if self.pca:
            pca = PCA(n_components=20)
            pca.fit(X_train_scaled)
            X_train_scaled = pca.transform(X_train_scaled)
            X_test_scaled = pca.transform(X_test_scaled)

        if self.smote:
            sm = SMOTE(random_state=42)
            X_train_scaled, y_train = sm.fit_resample(X_train_scaled, y_train)
            X_test_scaled, y_test = sm.fit_resample(X_test_scaled, y_test)

        self.model.fit(X_train_scaled, y_train)
        preds = self.model.predict(X_test_scaled)
        rmse = math.sqrt(mean_squared_error(preds, y_test))
        print(f"Accuracy: {accuracy_score(preds, y_test)}")
        print(f"Recall: {recall_score(preds, y_test)}")
        print(f"Precision: {precision_score(preds, y_test)}")
        print(f"RMSE: {rmse}")

    def kfolds_cv(self):
        kfold = KFold(n_splits=5, shuffle=True)
        accuracies = []
        recall = []
        precision = []
        rmse = []

        for train_index, test_index in kfold.split(self.features):
            self.model.fit(self.features[train_index], self.target[train_index])
            preds = self.model.predict(self.features[test_index])
            y_test = self.target[test_index]
            rmse_scores = math.sqrt(mean_squared_error(preds, y_test))
            accuracies.append(accuracy_score(y_test, preds))
            recall.append(recall_score(y_test, preds))
            precision.append(precision_score(y_test, preds))
            rmse.append(rmse_scores)

        print(f"Accuracy: {np.average(accuracies)}")
        print(f"Recall: {np.average(recall)}")
        print(f"Precision: {np.average(precision)}")
        print(f"RMSE: {np.average(rmse)}")

    def feature_importances(self):
        feat_importances = pd.Series(self.model.feature_importances_, index=self.features.columns)
        feat_importances.nlargest(20).plot(kind='barh')
