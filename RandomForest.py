import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from collections import defaultdict
from sklearn.random_projection import SparseRandomProjection, GaussianRandomProjection
from sklearn import preprocessing
from itertools import product
from sklearn.metrics.pairwise import pairwise_distances
from scipy.linalg import pinv
import scipy.sparse as sps
from sklearn.base import TransformerMixin,BaseEstimator
from sklearn.ensemble import RandomForestClassifier

def get_dataset1_from_csv(file_path):
    data_set = pd.read_csv(file_path, na_values='N/A',
                           dtype={'Number': int, 'carat': float, 'color': str, 'cut': str, 'clarity': str,
                                  'depth': float, 'table': float,
                                  'price': float, 'x': float,
                                  'y': float, 'z': float})
    data_set = data_set.fillna(0)
    header = ["carat", "color", "clarity", "depth",
              "table", "price", "x", "y", "z"]

    x_temp = data_set[header]
    y_temp = data_set['cut']
    x_ = x_temp.values.tolist()
    y_ = y_temp.values.tolist()
    x = np.asarray(x_)
    y = np.asarray(y_)
    return x, y, header


def get_dataset2_from_csv(file_path):
    data_set = pd.read_csv(file_path, na_values='N/A',
                           dtype={'ID': int, 'LIMIT_BAL': int,
                                  'SEX': int, 'EDUCATION': int, 'MARRIAGE': int, 'AGE': int,
                                  'PAY_0': int, 'PAY_2': int, 'PAY_3': int, 'PAY_4': int, 'PAY_5': str, 'PAY_6': int,
                                  'BILL_AMT1': int, 'BILL_AMT2': int, 'BILL_AMT3': int, 'BILL_AMT4': int,
                                  'BILL_AMT5': int, 'BILL_AMT6': int,
                                  'PAY_AMT1': int, 'PAY_AMT2': int, 'PAY_AMT3': int, 'PAY_AMT4': int, 'PAY_AMT5': int,
                                  'PAY_AMT6': int,
                                  'default_payment_next_month': int})
    # data_set = data_set.fillna(0)
    # data_set['Product_Category_1'] = data_set['Product_Category_1'].astype('int')
    header = ['LIMIT_BAL', 'SEX',
              'EDUCATION', 'MARRIAGE', 'AGE',
              'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
              'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
              'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']

    x_temp = data_set[header]
    y_temp = data_set['default_payment_next_month']
    x_ = x_temp.values.tolist()
    y_ = y_temp.values.tolist()
    x = np.asarray(x_)
    y = np.asarray(y_)
    return x, y, header


def split_training_test_data(data, split_ratio):
    num_samples = len(data)
    training_sample_number = int(split_ratio * num_samples)
    training = data[:training_sample_number]
    testing = data[training_sample_number:]
    return training, testing


def return_data():
    X_all, Y_all, header = get_dataset1_from_csv(file_path='diamonds.csv')
    X_all2, Y_all2, header2 = get_dataset2_from_csv(file_path='UCI_Credit_Card.csv')

    # Transfer the string variable to numerRFl/ineger variables
    number_of_features = len(header) - 1
    number_of_features2 = len(header2) - 1

    le = preprocessing.LabelEncoder()
    for i in range(0, number_of_features):
        X_all[:, i] = le.fit_transform(X_all[:, i])
    X_all = X_all.astype(np.float)
    Y_all = le.fit_transform(Y_all)

    lt = preprocessing.MinMaxScaler()
    lt.fit(X_all)
    X_all = lt.transform(X_all)
    lt.fit(X_all2)
    X_all2 = lt.transform(X_all2)
    # for i in range(0, number_of_features2):
    #     X_all2[:, i] = le.fit_transform(X_all2[:, i])
    # split the dataset to training and validation

    X_train, X_test = split_training_test_data(X_all, split_ratio=0.9)  # test~3000
    Y_train, Y_test = split_training_test_data(Y_all, split_ratio=0.9)

    X_train2, X_test2 = split_training_test_data(X_all2, split_ratio=0.9)  # test = 1200
    Y_train2, Y_test2 = split_training_test_data(Y_all2, split_ratio=0.9)

    # return X_all2,Y_all2
    return X_train, X_test, Y_train, Y_test, header, X_train2, X_test2, Y_train2, Y_test2, header2

np.random.seed(0)

class ImportanceSelect(BaseEstimator, TransformerMixin):
    def __init__(self, model, n=1):
         self.model = model
         self.n = n
    def fit(self, *args, **kwargs):
         self.model.fit(*args, **kwargs)
         return self
    def transform(self, X):
         return X[:,self.model.feature_importances_.argsort()[::-1][:self.n]]

#1.import data
X_train, X_test, Y_train, Y_test, header, X_train2, X_test2,Y_train2, Y_test2, header2 = return_data()
X_train = np.asarray(X_train)
X_test = np.asarray(X_test)
X_train2 = np.asarray(X_train2)
X_test2 = np.asarray(X_test2)

#2.RF feature importance
#2.1 Diamond datav
rfc1 = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=5, n_jobs=7)
fs_Diamond_train = rfc1.fit(X_train, Y_train).feature_importances_

# tmp = pd.Series(np.sort(fs_Diamond_train)[::-1])
tmp = pd.Series(fs_Diamond_train)
tmp.to_csv('Diamond_RF_train_feature.csv')


#2.2 Credit card data
rfc2 = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=5, n_jobs=7)
fs_Credit_card_train = rfc2.fit(X_train2, Y_train2).feature_importances_

# tmp = pd.Series(np.sort(fs_Credit_card_train)[::-1])
tmp = pd.Series(fs_Credit_card_train)
tmp.to_csv('Credit_card2_RF_train_feature.csv')

#3.RF transformation
dim = 9
filtr = ImportanceSelect(rfc1, dim)

#3.1 Diamond Train
DiamondX2_train = filtr.fit_transform(X_train, Y_train)
Diamond_train = pd.DataFrame(np.hstack((DiamondX2_train, np.atleast_2d(Y_train).T)))
cols = list(range(Diamond_train.shape[1]))
cols[-1] = 'Class'
Diamond_train.columns = cols
Diamond_train.to_csv('Diamond_RF_train.csv')

#3.2 Diamond Test
DiamondX2_test = filtr.fit_transform(X_test, Y_test)
Diamond_test = pd.DataFrame(np.hstack((DiamondX2_test, np.atleast_2d(Y_test).T)))
cols = list(range(Diamond_test.shape[1]))
cols[-1] = 'Class'
Diamond_test.columns = cols
Diamond_test.to_csv('Diamond_RF_test.csv')

#3.3 Credit Card Train
dim = 23
filtr = ImportanceSelect(rfc2, dim)

CreditCardX2_train = filtr.fit_transform(X_train2, Y_train2)
CreditCard_train = pd.DataFrame(np.hstack((CreditCardX2_train, np.atleast_2d(Y_train2).T)))
cols = list(range(CreditCard_train.shape[1]))
cols[-1] = 'Class'
CreditCard_train.columns = cols
CreditCard_train.to_csv('CreditCard_RF_train.csv')

#3.4 Credit Card test
CreditCardX2_test = filtr.fit_transform(X_test2, Y_test2)
CreditCard_test = pd.DataFrame(np.hstack((CreditCardX2_test, np.atleast_2d(Y_test2).T)))
cols = list(range(CreditCard_test.shape[1]))
cols[-1] = 'Class'
CreditCard_test.columns = cols
CreditCard_test.to_csv('CreditCard_RF_test.csv')