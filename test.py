import numpy as np
import pandas as pd 
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def read_mat(url):
    data = sio.loadmat(url)
    return data

data1 = read_mat(r"datasets\RAF_ML\DBM_CNN.mat")
features = data1["DBM_CNN"]
print(features.shape)

X_scaler = StandardScaler()
x = X_scaler.fit_transform(features)

pca = PCA(n_components=200)
pca.fit(x)
y=pca.transform(x)
print(y.shape)


data = []
with open('datasets\RAF_ML\distribution.txt') as f:
    for line in f.readlines():
        temp = line.split()
        data.append(temp[1:])
label=np.zeros((len(data),6))
print(label.shape)
for i in range(len(data)):
    for j in range(6):
        label[i][j]=float(data[i][j])
#print(label)




sio.savemat('RAF_ML.mat', {'features':y,'labels':label})




