# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 15:51:37 2017

@author: jtay
"""

#%% Imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from matplotlib import cm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn import preprocessing


# Credit card https://toolbox.google.com/datasetsearch/search?query=credit%20card%20application%20data&docid=KOZH6IFbY%2FvtQ1FtAAAAAA%3D%3D
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

    # Transfer the string variable to numerical/ineger variables
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

nn_arch= [(50,50),(50,),(25,),(25,25),(100,25,100)]
nn_reg = [10**-x for x in range(1,5)]
out = './PCA/'
cmap = cm.get_cmap('Spectral')

np.random.seed(0)

#1.import data
X_train, X_test, Y_train, Y_test, header, X_train2, X_test2,Y_train2, Y_test2, header2 = return_data()

#2.PCA Fitting-get the Eigenvalues
clusters =  [1,2,3,4,5,6,7,8,9,10,15,20,25,30,35,40]

#2.1 Diamond Data
pca = PCA(random_state=5)
pca.fit(X_train)
#The amount of variance explained by each of the selected components.
#Equal to n_components largest eigenvalues of the covariance matrix of X
tmp = pd.Series(data = pca.explained_variance_)
tmp.to_csv('Diamond_component_variance.csv')

#2.2 Credit Carda data
pca = PCA(random_state=5)
pca.fit(X_train2)
tmp = pd.Series(data = pca.explained_variance_)
tmp.to_csv('Credit_card_component_variance.csv')


#3.PCA transformation
#3.1Diamond data
dim = 9
pca = PCA(n_components=dim,random_state=10)
#3.1.1 Training data
DiamondX2_train = pca.fit_transform(X_train)
Diamond2_train = pd.DataFrame(np.hstack((DiamondX2_train,np.atleast_2d(Y_train).T)))
cols1 = list(range(Diamond2_train.shape[1]))

cols1[-1] = 'Class'
Diamond2_train.columns = cols1
Diamond2_train.to_csv('Diamond_PCA_10_train.csv')

#3.1.2 test data
DiamondX2_test = pca.fit_transform(X_test)
Diamond2_test = pd.DataFrame(np.hstack((DiamondX2_test,np.atleast_2d(Y_test).T)))
cols2 = list(range(Diamond2_test.shape[1]))

cols2[-1] = 'Class'
Diamond2_test.columns = cols2
Diamond2_test.to_csv('Diamond_PCA_10_test.csv')

#3.2 Credit card data
dim = 23
pca = PCA(n_components=dim,random_state=10)

#train data
Credit_cardX2_train = pca.fit_transform(X_train2)
Credit_card2_train = pd.DataFrame(np.hstack((Credit_cardX2_train,np.atleast_2d(Y_train2).T)))
cols3 = list(range(Credit_card2_train.shape[1]))
cols3[-1] = 'Class'
Credit_card2_train.columns = cols3
Credit_card2_train.to_csv('Credit_card2_PCA_24_train.csv')

#test data
Credit_cardX2_test = pca.fit_transform(X_test2)
Credit_card2_test = pd.DataFrame(np.hstack((Credit_cardX2_test,np.atleast_2d(Y_test2).T)))
cols4 = list(range(Credit_card2_test.shape[1]))
cols4[-1] = 'Class'
Credit_card2_test.columns = cols4
Credit_card2_test.to_csv('Credit_card2_PCA_24_test.csv')

