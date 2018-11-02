#import libraries
import Import_Two_data_set
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
    print(Y.shape)
    print(clusterLabels.shape)
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
#1.import data
X_train, X_test, Y_train, Y_test, header, X_train2, X_test2,Y_train2, Y_test2, header2 = Import_Two_data_set.return_data()

# print(X_train.head(2))
# print(Y_train.head(2))

# %% Data for 1-3
SSE = defaultdict(dict)
ll = defaultdict(dict)
aic = defaultdict(dict)
bic = defaultdict(dict)
acc_ = defaultdict(lambda: defaultdict(dict))
adjMI = defaultdict(lambda: defaultdict(dict))
km = kmeans(random_state=13)
gmm = GMM(random_state=13)

# clusters = [2, 5, 10, 15, 20, 25, 30, 35, 40]
clusters = [1,2,3,4,5,6,7,8,9,10,15, 20, 25, 30, 35, 40]
st = clock()
for k in clusters:
    km.set_params(n_clusters=k)
    gmm.set_params(n_components=k)
    km.fit(X_train)
    gmm.fit(X_train)
    #km.score = Opposite of the value of X on the K-means objective.
    #         =Sum of distances of samples to their closest cluster center
    SSE[k]['Diamond'] = km.score(X_train)
    ll[k]['Diamond'] = gmm.score(X_train)

    aic[k]['Diamond']  = gmm.aic(X_train)
    bic[k]['Diamond']  = gmm.bic(X_train)

    #training accuracy
    acc_[k]['Diamond']['Kmeans'] = cluster_acc(Y_test, km.predict(X_test))
    acc_[k]['Diamond']['GMM'] = cluster_acc(Y_test, gmm.predict(X_test))
    #mutual information score
    adjMI[k]['Diamond']['Kmeans'] = ami(Y_test, km.predict(X_test))
    adjMI[k]['Diamond']['GMM'] = ami(Y_test, gmm.predict(X_test))

    km.fit(X_train2)
    gmm.fit(X_train2)
    SSE[k]['CreditCard'] = km.score(X_train2)
    ll[k]['CreditCard'] = gmm.score(X_train2)
    aic[k]['CreditCard'] = gmm.aic(X_train2)
    bic[k]['CreditCard'] = gmm.bic(X_train2)

    acc_[k]['CreditCard']['Kmeans'] = cluster_acc(Y_test2, km.predict(X_test2))
    acc_[k]['CreditCard']['GMM'] = cluster_acc(Y_test2, gmm.predict(X_test2))
    adjMI[k]['CreditCard']['Kmeans'] = ami(Y_test2, km.predict(X_test2))
    adjMI[k]['CreditCard']['GMM'] = ami(Y_test2, gmm.predict(X_test2))
    print('cluster: ',k,'Wall clock time', clock() - st)

SSE = (-pd.DataFrame(SSE)).T
SSE.rename(columns=lambda x: x + ' SSE ', inplace=True)
ll = pd.DataFrame(ll).T
ll.rename(columns=lambda x: x + ' log-likelihood', inplace=True)

aic = (-pd.DataFrame(aic)).T
aic.rename(columns=lambda x: x + ' aic ', inplace=True)
bic = pd.DataFrame(bic).T
bic.rename(columns=lambda x: x + ' bic', inplace=True)

acc_ = pd.DataFrame(acc_).T
acc_.rename(columns=lambda x: x + ' Accuracy', inplace=True)
adjMI = pd.DataFrame(adjMI).T
adjMI.rename(columns=lambda x: x + ' Adjusted Mutual Information', inplace=True)
#print(SSE)
# acc2 = pd.Panel(acc_)
# adjMI2 = pd.Panel(adjMI)
#
#out = './{}/'.format(sys.argv[1])

SSE.to_csv('SSE-Sum_of_square_error.csv')
ll.to_csv('LL-logliklihood.csv')
aic.to_csv('aic-Akaike information criterion.csv')
bic.to_csv('bic-Bayesian information criterion.csv')
acc_.to_csv('Diamond_CreditCard_acc.csv')
adjMI.to_csv('Diamond_CreditCard_adjMI.csv')

