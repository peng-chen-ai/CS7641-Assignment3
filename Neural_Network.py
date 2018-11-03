from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import optimizers
from keras import backend as k
from keras.utils import to_categorical
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from keras.layers.advanced_activations import LeakyReLU, PReLU
from sklearn.utils import class_weight
import numpy as np
import pandas as pd
from sklearn import preprocessing
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
import numpy as np
np.random.seed(13)
# fix random seed for reproducibility
np.random.seed(7)
import time
from datetime import datetime
#setup rendering
start_time = time.time()
start = datetime.now()
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
                           dtype={ '1': int,
                                  '2': int, '3': int, '4': int, '5': int,
                                  '6': int, '7': int, '8': int, '9': int, '10': str, '11': int,
                                  '12': int, '13': int, '14': int, '15': int,
                                  '16': int, '17': int,
                                  '18': int, '19': int, '20': int, '21': int, '22': int,
                                  '23': int,
                                  'Class': int})
    # data_set = data_set.fillna(0)
    # data_set['Product_Category_1'] = data_set['Product_Category_1'].astype('int')
    header = ['1', '2', '3', '4', '5', '6', '7', '8','9','10','11','12',
                                         '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23']

    x_temp = data_set[header]
    y_temp = data_set['Class']
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
    X_all2, Y_all2, header2 = get_dataset2_from_csv(file_path='UCI_Credit_Card_rename.csv')

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

def Hubert_loss(y_true, y_pred):
    err = y_pred -y_true
    return k.mean(k.sqrt(1+k.square(err))-1, axis = -1)

def read_Dimension_reduction_data(col,File_Path_train ,File_Path_test):
    #col = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
    RD_DATA_train = pd.read_csv(File_Path_train)
    RD_DATA_trainX = RD_DATA_train[col]
    RD_DATA_trainY = RD_DATA_train['Class']

    RD_DATA_test = pd.read_csv(File_Path_test)
    RD_DATA_testX = RD_DATA_test[col]
    RD_DATA_testY = RD_DATA_test['Class']

    RD_DATA_trainX_ = np.asarray(RD_DATA_trainX)
    RD_DATA_trainY_ = np.asarray(RD_DATA_trainY)
    RD_DATA_testX_ = np.asarray(RD_DATA_testX)
    RD_DATA_testY_ = np.asarray(RD_DATA_testY)
    #print(RD_DATA_trainX.head(2), RD_DATA_trainY.head(2), RD_DATA_testX.head(2), RD_DATA_testY.head(2))
    return RD_DATA_trainX_, RD_DATA_trainY_, RD_DATA_testX_, RD_DATA_testY_, len(col)


#Experiment - Learning CUrve
def Experiment_learning_curve(xtrain, ytrain, xtest, ytest,
                              input_dimension,Predictor_type,
                              layer1_dense,hidden_layer_dense,output_layer_dense = 1,learnRate = 0.001,
                              num_of_component= [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]):
                              #   num_of_component = [1,23]):
    RMSD_List_E = []
    for num_col in num_of_component:
        # sub_col_temp = num_of_component[:num_col]
        # sub_col = [str(item) for item in sub_col_temp]
        print(num_col)
        X_nk_train = xtrain[:,:num_col]
        xtest_ = xtest[:,:num_col]
        input_dimension = num_col

        X_nk_train = X_nk_train[0:7000]
        Y_nk_train = ytrain[0:7000]

        #class_weights = {0: 1., 1: 1}
        model_name = Sequential()
        model_name.add(Dense(layer1_dense, input_dim=input_dimension, activation='relu'))
        # model_name.add(LeakyReLU(alpha=0.003))
        model_name.add(Dense(hidden_layer_dense, activation='relu'))

        if Predictor_type == 'Binary':
            model_name.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
            adam = optimizers.Adam(lr=learnRate)
            model_name.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])  # 'rmsprop'
            model_name.fit(X_nk_train, Y_nk_train, verbose=0,  # class_weight=class_weights,
                           epochs=300, batch_size=20)
        else:
            model_name.add(Dense(output_layer_dense, kernel_initializer='normal', activation='sigmoid'))
            adam = optimizers.Adam(lr=learnRate)
            model_name.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
            Y_nk_train_dummy = to_categorical(Y_nk_train)
            model_name.fit(X_nk_train, Y_nk_train_dummy, verbose=0,# class_weight=class_weights,
                           epochs=300, batch_size=20)


        y_predict_test = model_name.predict_classes(xtest_)
        y_predict_train = model_name.predict_classes(X_nk_train)

        RMSD_train = accuracy_score(y_predict_train, Y_nk_train)
        RMSD_test = accuracy_score(y_predict_test, ytest)
        RMSD_List_E.append([num_col,round(RMSD_test, 2), round(RMSD_train, 2)])
        print([num_col,round(RMSD_test, 2), round(RMSD_train, 2)])

    return RMSD_List_E


#1.import data-Credit Card from 4 Dimension reduction transformed data
# 1.0 Original Data
X_train, X_test, Y_train, Y_test, header, X_train2, X_test2,Y_train2, Y_test2, header2 = return_data()

#1.1 Principle Component
Credit_PCA_TrainX, Credit_PCA_TrainY, Credit_PCA_TestX,Credit_PCA_TestY, dimension_PCA = \
    read_Dimension_reduction_data(col = ['1', '2', '3', '4', '5', '6', '7', '8','9','10','11','12',
                                         '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23'],
                                  File_Path_train = 'Credit_card2_PCA_24_train_.csv' ,
                                  File_Path_test = 'Credit_card2_PCA_24_test_.csv')

#1.2 Independent Component
Credit_ICA_TrainX, Credit_ICA_TrainY, Credit_ICA_TestX,Credit_ICA_TestY,dimension_ICA = \
    read_Dimension_reduction_data(col = ['1', '2', '3', '4', '5', '6', '7', '8','9','10','11','12',
                                         '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23'],
                                  File_Path_train = 'Credit_card2_ICA_train.csv' ,
                                  File_Path_test = 'Credit_card2_ICA_test.csv')
#1.3 Randomized Projection
Credit_RP_TrainX, Credit_RP_TrainY, Credit_RP_TestX,Credit_RP_TestY,dimension_RP = \
    read_Dimension_reduction_data(col = ['1', '2', '3', '4', '5', '6', '7', '8','9','10','11','12',
                                         '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23'],
                                  File_Path_train = 'Credit_card2_RP_train.csv' ,
                                  File_Path_test = 'Credit_card2_RP_test.csv')
#1.4 Random Forest
Credit_RF_TrainX, Credit_RF_TrainY, Credit_RF_TestX,Credit_RF_TestY,dimension_RF = \
    read_Dimension_reduction_data(col = ['1', '2', '3', '4', '5', '6', '7', '8','9','10','11','12',
                                         '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23'],
                                  File_Path_train = 'CreditCard_RF_train.csv' ,
                                  File_Path_test = 'CreditCard_RF_test.csv')

#2.Learning Curve Experiment
#2.1 Original Data

start_time = time.time()
RMSD_List_Original_CreditCard = Experiment_learning_curve(xtrain = X_train2, ytrain = Y_train2,
                                                      xtest = X_test2, ytest = Y_test2, input_dimension =len(header2),
                                                     Predictor_type = 'Binary', layer1_dense = 50,
                                                     hidden_layer_dense = 40,output_layer_dense = 1)
end_time1 = time.time()-start_time
print("CreditCard-Original data time:", end_time1,RMSD_List_Original_CreditCard)

#2.2 PCA Transformed DataData
start_time = time.time()
RMSD_List_PCA_CreditCard = Experiment_learning_curve(xtrain = Credit_PCA_TrainX, ytrain = Credit_PCA_TrainY,
                                                     xtest = Credit_PCA_TestX, ytest = Credit_PCA_TestY,
                                                     input_dimension =dimension_PCA,Predictor_type='Binary',layer1_dense = 50,
                                                     hidden_layer_dense = 40,output_layer_dense = 1 )
end_time2 = time.time()-start_time
print("Credit_card-PCA time:", end_time2,RMSD_List_PCA_CreditCard)

#2.3.ICA Transformed DataData
start_time = time.time()
RMSD_List_ICA_CreditCard = Experiment_learning_curve(xtrain = Credit_ICA_TrainX, ytrain = Credit_ICA_TrainY,
                                                     xtest = Credit_ICA_TestX, ytest = Credit_ICA_TestY,
                                                     input_dimension =dimension_ICA,Predictor_type='Binary',layer1_dense = 50,
                                                     hidden_layer_dense = 40,output_layer_dense = 1 )
end_time3 = time.time()-start_time
print("Diamonds-ICA time:", end_time3, RMSD_List_ICA_CreditCard)

#2.4 RP Transformed DataData
start_time = time.time()
RMSD_List_RP_CreditCard = Experiment_learning_curve(xtrain = Credit_RP_TrainX, ytrain = Credit_RP_TrainY,
                                                     xtest = Credit_RP_TestX, ytest = Credit_RP_TestY,
                                                     input_dimension =dimension_RP,Predictor_type='Binary',layer1_dense = 50,
                                                     hidden_layer_dense = 40,output_layer_dense = 1 )
end_time4 = time.time()-start_time
print("Credit_card-RP time:", end_time4,RMSD_List_RP_CreditCard)

#2.5 RF Transformed DataData
start_time = time.time()
RMSD_List_RF_CreditCard = Experiment_learning_curve(xtrain = Credit_RF_TrainX, ytrain = Credit_RF_TrainY,
                                                     xtest = Credit_RF_TestX, ytest = Credit_RF_TestY,
                                                     input_dimension =dimension_RF,Predictor_type='Binary',layer1_dense = 50,
                                                     hidden_layer_dense = 40,output_layer_dense = 1 )
end_time5 = time.time()-start_time
print("Credit_card-RF time:", end_time5, RMSD_List_RF_CreditCard)



