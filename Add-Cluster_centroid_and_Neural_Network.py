#import libraries
import numpy as np
np.random.seed(13)
import pandas as pd
from time import clock

from sklearn.cluster import KMeans as kmeans

#1. Read in XX(Dimensionality Reduction) Transformed Credit Card data
Creditcard_col = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23']
#1.1import PCA data
PCA_CreditCard_train = pd.read_csv('Credit_card2_PCA_24_train_.csv')
PCA_CreditCard_trainX = PCA_CreditCard_train[Creditcard_col]
PCA_CreditCard_trainY = PCA_CreditCard_train['Class']

PCA_CreditCard_Test = pd.read_csv('Credit_card2_PCA_24_test_.csv')
PCA_CreditCard_TestX = PCA_CreditCard_Test[Creditcard_col]
PCA_CreditCard_TestY = PCA_CreditCard_Test['Class']

#1.2.Import ICA Data
ICA_CreditCard_train = pd.read_csv('Credit_card2_ICA_train.csv')
ICA_CreditCard_trainX = ICA_CreditCard_train[Creditcard_col]
ICA_CreditCard_trainY = ICA_CreditCard_train['Class']

ICA_CreditCard_Test = pd.read_csv('Credit_card2_ICA_test.csv')
ICA_CreditCard_TestX = ICA_CreditCard_Test[Creditcard_col]
ICA_CreditCard_TestY = ICA_CreditCard_Test['Class']

#1.3.Import Random Projection Data
RP_CreditCard_train = pd.read_csv('Credit_card2_RP_train.csv')
RP_CreditCard_trainX = RP_CreditCard_train[Creditcard_col]
RP_CreditCard_trainY = RP_CreditCard_train['Class']

RP_CreditCard_Test = pd.read_csv('Credit_card2_RP_test.csv')
RP_CreditCard_TestX = RP_CreditCard_Test[Creditcard_col]
RP_CreditCard_TestY = RP_CreditCard_Test['Class']

#1.4.Import Random Forest Data
RF_CreditCard_train = pd.read_csv('CreditCard_RF_train.csv')
RF_CreditCard_trainX = RF_CreditCard_train[Creditcard_col]
RF_CreditCard_trainY = RF_CreditCard_train['Class']

RF_CreditCard_Test = pd.read_csv('CreditCard_RF_test.csv')
RF_CreditCard_TestX = RF_CreditCard_Test[Creditcard_col]
RF_CreditCard_TestY = RF_CreditCard_Test['Class']

def clustering_append_centroid( X_data,algorithm,clusters = [ 2, 5, 10, 25]):
    X_data2_ = X_data[['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23']]
    km = kmeans(random_state=15)
    for k in clusters:
        km.set_params(n_clusters=k)
        km = km.fit(X_data2_)
        X_data2_['Cluster'+str(k)] = km.labels_
        #X_train2_.to_csv(str(algorithm)+'Transformed-Add-Cluster-2-5-10-25-Credit_card_train.csv')
    return X_data2_

PCA_Credit_card_train_ADD_Cluter2_5_10_25 = clustering_append_centroid(X_data=PCA_CreditCard_trainX,algorithm='PCA')
ICA_Credit_card_train_ADD_Cluter2_5_10_25 = clustering_append_centroid(X_data=ICA_CreditCard_trainX,algorithm='ICA')
RP_Credit_card_train_ADD_Cluter2_5_10_25 = clustering_append_centroid(X_data=RP_CreditCard_trainX,algorithm='RP')
RF_Credit_card_train_ADD_Cluter2_5_10_25 = clustering_append_centroid(X_data=RF_CreditCard_trainX,algorithm='RF')

PCA_Credit_card_Test_ADD_Cluter2_5_10_25 = clustering_append_centroid(X_data=PCA_CreditCard_TestX,algorithm='PCA')
ICA_Credit_card_Test_ADD_Cluter2_5_10_25 = clustering_append_centroid(X_data=ICA_CreditCard_TestX,algorithm='ICA')
RP_Credit_card_Test_ADD_Cluter2_5_10_25 = clustering_append_centroid(X_data=RP_CreditCard_TestX,algorithm='RP')
RF_Credit_card_Test_ADD_Cluter2_5_10_25 = clustering_append_centroid(X_data=RF_CreditCard_TestX,algorithm='RF')
# print('PCA',PCA_Credit_card_Test_ADD_Cluter2_5_10_25.head(2))
# print('ICA',ICA_Credit_card_Test_ADD_Cluter2_5_10_25.head(2))
# print('RP',RP_Credit_card_Test_ADD_Cluter2_5_10_25.head(2))
# print('RF',RF_Credit_card_Test_ADD_Cluter2_5_10_25.head(2))


#2.Neural Network Training and Training data accuracy
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

#Experiment - Learning CUrve
def Experiment_NN(xtrain, ytrain,xtest,ytest,
                              cluster_feature,Predictor_type = 'Binary',
                              layer1_dense = 50, hidden_layer_dense = 40,output_layer_dense = 1,learnRate = 0.001,
                              num_of_component= ['1','2','3','4','5','6','7','8','9','10','11','12',
                                                 '13','14','15','16','17','18','19','20','21','22','23']):
                              #   num_of_component = [1,23]):
    RMSD_List_E = []
    if len(cluster_feature) > 0:
        num_of_component.append(cluster_feature)

    X_nk_train = xtrain[num_of_component]
    X_nk_train = np.asarray(X_nk_train)
    ytrain = np.asarray(ytrain)
    X_nk_test = xtest[num_of_component]
    X_nk_test= np.asarray(X_nk_test)
    Y_nk_test = np.asarray(ytest)

    input_dimension = len(num_of_component)

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

    y_predict_train = model_name.predict_classes(X_nk_train)
    y_predict_test = model_name.predict_classes(X_nk_test)

    RMSD_train = accuracy_score(y_predict_train, Y_nk_train)
    RMSD_test = accuracy_score(y_predict_test, Y_nk_test)

    RMSD_List_E.append([cluster_feature, round(RMSD_train, 2),round(RMSD_test,2)])

    return RMSD_List_E

#2.1 PCA -Accuracy
PCA_Accuracy = []
PCA_NN_Add_cluster2 = Experiment_NN(xtrain = PCA_Credit_card_train_ADD_Cluter2_5_10_25, ytrain = PCA_CreditCard_trainY,
                                    xtest = PCA_Credit_card_Test_ADD_Cluter2_5_10_25, ytest = PCA_CreditCard_TestY,
                                    cluster_feature = 'Cluster2')
PCA_NN_Add_cluster5 = Experiment_NN(xtrain = PCA_Credit_card_train_ADD_Cluter2_5_10_25, ytrain = PCA_CreditCard_trainY,
                                    xtest=PCA_Credit_card_Test_ADD_Cluter2_5_10_25, ytest=PCA_CreditCard_TestY,
                                    cluster_feature = 'Cluster5')
PCA_NN_Add_cluster10 = Experiment_NN(xtrain = PCA_Credit_card_train_ADD_Cluter2_5_10_25, ytrain = PCA_CreditCard_trainY,
                                     xtest=PCA_Credit_card_Test_ADD_Cluter2_5_10_25, ytest=PCA_CreditCard_TestY,
                                    cluster_feature = 'Cluster10')
PCA_NN_Add_cluster25 = Experiment_NN(xtrain = PCA_Credit_card_train_ADD_Cluter2_5_10_25, ytrain = PCA_CreditCard_trainY,
                                    xtest=PCA_Credit_card_Test_ADD_Cluter2_5_10_25, ytest=PCA_CreditCard_TestY,
                                    cluster_feature = 'Cluster25')
PCA_Accuracy.append(PCA_NN_Add_cluster2)
PCA_Accuracy.append(PCA_NN_Add_cluster5)
PCA_Accuracy.append(PCA_NN_Add_cluster10)
PCA_Accuracy.append(PCA_NN_Add_cluster25)
print('PCA-Training+Test',PCA_Accuracy)

#2.2 ICA -Accuracy
ICA_Accuracy = []
ICA_NN_Add_cluster2 = Experiment_NN(xtrain = ICA_Credit_card_train_ADD_Cluter2_5_10_25, ytrain = ICA_CreditCard_trainY,
                                    xtest=ICA_Credit_card_Test_ADD_Cluter2_5_10_25, ytest=ICA_CreditCard_TestY,
                                    cluster_feature = 'Cluster2')
ICA_NN_Add_cluster5 = Experiment_NN(xtrain = ICA_Credit_card_train_ADD_Cluter2_5_10_25, ytrain = ICA_CreditCard_trainY,
                                    xtest=ICA_Credit_card_Test_ADD_Cluter2_5_10_25, ytest=ICA_CreditCard_TestY,
                                    cluster_feature = 'Cluster5')
ICA_NN_Add_cluster10 = Experiment_NN(xtrain = ICA_Credit_card_train_ADD_Cluter2_5_10_25, ytrain = ICA_CreditCard_trainY,
                                     xtest=ICA_Credit_card_Test_ADD_Cluter2_5_10_25, ytest=ICA_CreditCard_TestY,
                                    cluster_feature = 'Cluster10')
ICA_NN_Add_cluster25 = Experiment_NN(xtrain = ICA_Credit_card_train_ADD_Cluter2_5_10_25, ytrain = ICA_CreditCard_trainY,
                                     xtest=ICA_Credit_card_Test_ADD_Cluter2_5_10_25, ytest=ICA_CreditCard_TestY,
                                    cluster_feature = 'Cluster25')
ICA_Accuracy.append(ICA_NN_Add_cluster2)
ICA_Accuracy.append(ICA_NN_Add_cluster5)
ICA_Accuracy.append(ICA_NN_Add_cluster10)
ICA_Accuracy.append(ICA_NN_Add_cluster25)
print('ICA-Training+Test',ICA_Accuracy)

#2.3 RP -Accuracy
RP_Accuracy = []
RP_NN_Add_cluster2 = Experiment_NN(xtrain = RP_Credit_card_train_ADD_Cluter2_5_10_25, ytrain = RP_CreditCard_trainY,
                                   xtest=RP_Credit_card_Test_ADD_Cluter2_5_10_25, ytest=RP_CreditCard_TestY,
                                    cluster_feature = 'Cluster2')
RP_NN_Add_cluster5 = Experiment_NN(xtrain = RP_Credit_card_train_ADD_Cluter2_5_10_25, ytrain = RP_CreditCard_trainY,
                                   xtest=RP_Credit_card_Test_ADD_Cluter2_5_10_25, ytest=RP_CreditCard_TestY,
                                    cluster_feature = 'Cluster5')
RP_NN_Add_cluster10 = Experiment_NN(xtrain = RP_Credit_card_train_ADD_Cluter2_5_10_25, ytrain = RP_CreditCard_trainY,
                                    xtest=RP_Credit_card_Test_ADD_Cluter2_5_10_25, ytest=RP_CreditCard_TestY,
                                    cluster_feature = 'Cluster10')
RP_NN_Add_cluster25 = Experiment_NN(xtrain = RP_Credit_card_train_ADD_Cluter2_5_10_25, ytrain = RP_CreditCard_trainY,
                                    xtest=RP_Credit_card_Test_ADD_Cluter2_5_10_25, ytest=RP_CreditCard_TestY,
                                    cluster_feature = 'Cluster25')
RP_Accuracy.append(RP_NN_Add_cluster2)
RP_Accuracy.append(RP_NN_Add_cluster5)
RP_Accuracy.append(RP_NN_Add_cluster10)
RP_Accuracy.append(RP_NN_Add_cluster25)
print('RP-Training+Test',RP_Accuracy)

#2.4 RF -Accuracy
RF_Accuracy = []
RF_NN_Add_cluster2 = Experiment_NN(xtrain = RF_Credit_card_train_ADD_Cluter2_5_10_25, ytrain = RF_CreditCard_trainY,
                                    xtest=RF_Credit_card_Test_ADD_Cluter2_5_10_25, ytest=RF_CreditCard_TestY,
                                    cluster_feature = 'Cluster2')
RF_NN_Add_cluster5 = Experiment_NN(xtrain = RF_Credit_card_train_ADD_Cluter2_5_10_25, ytrain = RF_CreditCard_trainY,
                                   xtest=RF_Credit_card_Test_ADD_Cluter2_5_10_25, ytest=RF_CreditCard_TestY,
                                    cluster_feature = 'Cluster5')
RF_NN_Add_cluster10 = Experiment_NN(xtrain = RF_Credit_card_train_ADD_Cluter2_5_10_25, ytrain = RF_CreditCard_trainY,
                                    xtest=RF_Credit_card_Test_ADD_Cluter2_5_10_25, ytest=RF_CreditCard_TestY,
                                    cluster_feature = 'Cluster10')
RF_NN_Add_cluster25 = Experiment_NN(xtrain = RF_Credit_card_train_ADD_Cluter2_5_10_25, ytrain = RF_CreditCard_trainY,
                                    xtest=RF_Credit_card_Test_ADD_Cluter2_5_10_25, ytest=RF_CreditCard_TestY,
                                    cluster_feature = 'Cluster25')
RF_Accuracy.append(RF_NN_Add_cluster2)
RF_Accuracy.append(RF_NN_Add_cluster5)
RF_Accuracy.append(RF_NN_Add_cluster10)
RF_Accuracy.append(RF_NN_Add_cluster25)
print('RF-Training+Test',RF_Accuracy)

#3. Use the Clustering Feature only for NN training and get the training accuracy
PCA_NN_Add_cluster2_5_10_25 = Experiment_NN(xtrain = PCA_Credit_card_train_ADD_Cluter2_5_10_25, ytrain = PCA_CreditCard_trainY,
                                            xtest=PCA_Credit_card_Test_ADD_Cluter2_5_10_25, ytest=PCA_CreditCard_TestY,
                                    cluster_feature = 'Cluster2',num_of_component = ['Cluster5','Cluster10','Cluster25'])
ICA_NN_Add_cluster2_5_10_25 = Experiment_NN(xtrain = ICA_Credit_card_train_ADD_Cluter2_5_10_25, ytrain = ICA_CreditCard_trainY,
                                            xtest=ICA_Credit_card_Test_ADD_Cluter2_5_10_25, ytest=ICA_CreditCard_TestY,
                                    cluster_feature = 'Cluster2',num_of_component = ['Cluster5','Cluster10','Cluster25'])
RP_NN_Add_cluster2_5_10_25 = Experiment_NN(xtrain = RP_Credit_card_train_ADD_Cluter2_5_10_25, ytrain = RP_CreditCard_trainY,
                                           xtest=RP_Credit_card_Test_ADD_Cluter2_5_10_25, ytest=RP_CreditCard_TestY,
                                    cluster_feature = 'Cluster2',num_of_component = ['Cluster5','Cluster10','Cluster25'])
RF_NN_Add_cluster2_5_10_25 = Experiment_NN(xtrain = RF_Credit_card_train_ADD_Cluter2_5_10_25, ytrain = RF_CreditCard_trainY,
                                           xtest=RF_Credit_card_Test_ADD_Cluter2_5_10_25, ytest=RF_CreditCard_TestY,
                                    cluster_feature = 'Cluster2',num_of_component = ['Cluster5','Cluster10','Cluster25'])
print('PCA_NN_Add_cluster2_5_10_25',PCA_NN_Add_cluster2_5_10_25)
print('ICA_NN_Add_cluster2_5_10_25',ICA_NN_Add_cluster2_5_10_25)
print('RP_NN_Add_cluster2_5_10_25',RP_NN_Add_cluster2_5_10_25)
print('RF_NN_Add_cluster2_5_10_25',RF_NN_Add_cluster2_5_10_25)

PCA_NN_Add_cluster25 = Experiment_NN(xtrain = PCA_Credit_card_train_ADD_Cluter2_5_10_25, ytrain = PCA_CreditCard_trainY,
                                            xtest=PCA_Credit_card_Test_ADD_Cluter2_5_10_25, ytest=PCA_CreditCard_TestY,
                                     cluster_feature= [], num_of_component = ['Cluster25'])
ICA_NN_Add_cluster10 = Experiment_NN(xtrain = ICA_Credit_card_train_ADD_Cluter2_5_10_25, ytrain = ICA_CreditCard_trainY,
                                            xtest=ICA_Credit_card_Test_ADD_Cluter2_5_10_25, ytest=ICA_CreditCard_TestY,
                                     cluster_feature=[], num_of_component = ['Cluster10'])
RP_NN_Add_cluster2 = Experiment_NN(xtrain = RP_Credit_card_train_ADD_Cluter2_5_10_25, ytrain = RP_CreditCard_trainY,
                                           xtest=RP_Credit_card_Test_ADD_Cluter2_5_10_25, ytest=RP_CreditCard_TestY,
                                   cluster_feature=[], num_of_component = ['Cluster2'])
RF_NN_Add_cluster25 = Experiment_NN(xtrain = RF_Credit_card_train_ADD_Cluter2_5_10_25, ytrain = RF_CreditCard_trainY,
                                           xtest=RF_Credit_card_Test_ADD_Cluter2_5_10_25, ytest=RF_CreditCard_TestY,
                                    cluster_feature=[],  num_of_component = ['Cluster25'])
print('PCA_NN_Add_cluster5_25',PCA_NN_Add_cluster25)
print('ICA_NN_Add_cluster2_10',ICA_NN_Add_cluster10)
print('RP_NN_Add_cluster2_10',RP_NN_Add_cluster2)
print('RF_NN_Add_cluster10_25',RF_NN_Add_cluster25)
