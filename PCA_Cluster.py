#import libraries
import numpy as np
np.random.seed(13)
import pandas as pd
from sklearn.manifold import TSNE
from time import clock
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans as kmeans
from sklearn.mixture import GaussianMixture as GMM
from collections import defaultdict
from sklearn.metrics import adjusted_mutual_info_score as ami
from collections import Counter
from collections import defaultdict
from sklearn.metrics import accuracy_score as acc
import sys
def cluster_acc(Y, clusterLabels):
    assert (Y.shape == clusterLabels.shape)
    pred = np.empty_like(Y)
    for label in set(clusterLabels):
        mask = clusterLabels == label
        sub = Y[mask]
        target = Counter(sub).most_common(1)[0][0]
        pred[mask] = target
    # assert max(pred) == max(Y)
    #    assert min(pred) == min(Y)
    return acc(Y, pred)

class myGMM(GMM):
    def transform(self, X):
        return self.predict_proba(X)
#1.import PCA data
#1.1 Diamond data
diamond_col = ['1','2','3','4','5','6','7','8','9']
PCA_diamond_train = pd.read_csv('Diamond_PCA_10_train_.csv')
PCA_diamond_trainX = PCA_diamond_train[diamond_col]
PCA_diamond_trainY = PCA_diamond_train['Class']

PCA_diamond_test = pd.read_csv('Diamond_PCA_10_test_.csv')
PCA_diamond_testX = PCA_diamond_test[diamond_col]
PCA_diamond_testY = PCA_diamond_test['Class']

#1.2 Credit Card data
Creditcard_col = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23']
PCA_CreditCard_train = pd.read_csv('Credit_card2_PCA_24_train_.csv')
PCA_CreditCard_trainX = PCA_CreditCard_train[Creditcard_col]
PCA_CreditCard_trainY = PCA_CreditCard_train['Class']

PCA_CreditCard_test = pd.read_csv('Credit_card2_PCA_24_test_.csv')
PCA_CreditCard_testX = PCA_CreditCard_test[Creditcard_col]
PCA_CreditCard_testY = PCA_CreditCard_test['Class']

#2. Fit clustering model
acc_ = defaultdict(lambda: defaultdict(dict))

km = kmeans(random_state=13)
gmm = GMM(random_state=13)

def k_mean_and_EM(train_data_x, test_data_x,train_data_y,test_data_y,k,num_of_component,data_set):
    st = clock()
    acc_ = defaultdict(lambda: defaultdict(dict))
    for num_col in num_of_component:
        sub_col_temp = num_of_component[:num_col]
        sub_col = [str(item) for item in sub_col_temp]
        print(sub_col)
        train_data_x_ = train_data_x[sub_col]
        test_data_x_ = test_data_x[sub_col]

        train_data_x_ = np.asarray(train_data_x_)
        train_data_y = np.asarray(train_data_y)
        test_data_x_ = np.asarray(test_data_x_)
        test_data_y = np.asarray(test_data_y)

        km.set_params(n_clusters=k)
        gmm.set_params(n_components=k)
        km.fit(train_data_x_)
        gmm.fit(train_data_x_)

        #test accuracy
        acc_[num_col][data_set]['Kmeans'] = cluster_acc(test_data_y, km.predict(test_data_x_))
        acc_[num_col][data_set]['GMM'] = cluster_acc(test_data_y, gmm.predict(test_data_x_))
    
    return acc_

#2.1 Diamond data
acc_diamond_pca_= k_mean_and_EM(train_data_x = PCA_diamond_trainX, test_data_x = PCA_diamond_testX,
                                train_data_y = PCA_diamond_trainY ,test_data_y = PCA_diamond_testY,
                                k = 6,num_of_component = [1,2,3,4,5,6,7,8,9],data_set = 'Diamond')
acc_diamond_pca = pd.DataFrame(acc_diamond_pca_).T
acc_diamond_pca.rename(columns=lambda x: x + ' Accuracy', inplace=True)
acc_diamond_pca.to_csv('Diamond_acc_pca.csv')

#2.2 Credit Card data
acc_CreditCard_pca_ =k_mean_and_EM(train_data_x = PCA_CreditCard_trainX, test_data_x = PCA_CreditCard_testX,
                                   train_data_y = PCA_CreditCard_trainY ,test_data_y = PCA_CreditCard_testY,
                                   k = 3,num_of_component = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23],
                                   data_set = 'CreditCard')
acc_CreditCard_pca = pd.DataFrame(acc_CreditCard_pca_).T
acc_CreditCard_pca.rename(columns=lambda x: x + ' Accuracy', inplace=True)
acc_CreditCard_pca.to_csv('CreditCard_acc_pca.csv')

# 3 Clustering analysis on newly transformed data
def clustering(X_train,X_test, X_train2,X_test2):

    X_train_ = X_train[['1','2','3','4','5','6']]
    X_test_ = X_test[['1','2','3','4','5','6']]
    X_train2_ = X_train2[['1', '2', '3', '4', '5', '6', '7', '8']]
    X_test2_ = X_test2[['1', '2', '3', '4', '5', '6', '7', '8']]

    # X_train = np.asarray(X_train)
    # X_test = np.asarray(X_test)
    # X_train2 = np.asarray(X_train2)
    # X_test2 = np.asarray(X_test2)

    clusters = [1,2,3,4,5,6,7,8,9,10,15,20,25,30,35,40]
    st = clock()
    SSE = defaultdict(dict)
    ll = defaultdict(dict)
    km = kmeans(random_state=15)
    gmm = GMM(random_state=50)
    for k in clusters:
        km.set_params(n_clusters=k)
        gmm.set_params(n_components=k)
        km.fit(X_train_)
        gmm.fit(X_train_)
        #km.score = Opposite of the value of X on the K-means objective.
        #         =Sum of distances of samples to their closest cluster center
        SSE[k]['Diamond'] = km.score(X_test_)
        ll[k]['Diamond'] = gmm.score(X_test_)

        km.fit(X_train2_)
        gmm.fit(X_train2_)
        SSE[k]['CreditCard'] = km.score(X_test2_)
        ll[k]['CreditCard'] = gmm.score(X_test2_)

    return SSE,ll
CreditCard_Diamond_SSE_PCA_, CreditCard_Diamond_PCA_LL_ \
    = clustering(X_train = PCA_diamond_trainX, X_test = PCA_diamond_testX,
                 X_train2= PCA_CreditCard_trainX, X_test2= PCA_CreditCard_testX)

CreditCard_Diamond_SSE_PCA = (-pd.DataFrame(CreditCard_Diamond_SSE_PCA_)).T
CreditCard_Diamond_SSE_PCA.rename(columns=lambda x: x + ' SSE ', inplace=True)
CreditCard_Diamond_PCA_LL = pd.DataFrame(CreditCard_Diamond_PCA_LL_).T
CreditCard_Diamond_PCA_LL.rename(columns=lambda x: x + ' log-likelihood', inplace=True)

CreditCard_Diamond_SSE_PCA.to_csv('SSE-Sum_of_square_error-PCA.csv')
CreditCard_Diamond_PCA_LL.to_csv('LL-logliklihood-PCA.csv')


