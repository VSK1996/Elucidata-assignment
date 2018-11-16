import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from matplotlib import*

from sklearn.decomposition import PCA 


inten = pd.read_csv('g.csv')
meta = pd.read_csv('m.csv')


len(inten)

dta_ch = inten.iloc[:,2:]
dta_ch=dta_ch.replace({'ssssss': '39.0', 'hhhh' : '321.43'}, regex=True)

X=dta_ch.values
X=X.astype(float)
X[np.isnan(X)] = np.median(X[~np.isnan(X)])
X = X.T
X.shape

X_std = StandardScaler().fit_transform(X)
pca = PCA(n_components=2)
pca.fit(X_std)

y = pca.components_

w = pca.explained_variance_
b = 0.0
for i in range(len(w)):
    b = b+ float(w[i])
print (b)

Y = pca.fit_transform(X)

Y.shape

sample = np.array(meta['Time'])

x_min, x_max = Y[:, 0].min() - 10000, Y[:, 0].max() + 10000
y_min, y_max = Y[:, 1].min() - 10000, Y[:, 1].max() + 10000

plt.figure(figsize= (15,10))

plt.scatter(Y[:, 0], Y[:, 1], c=sample, cmap=plt.cm.Set1,
            edgecolor='k')
plt.xlabel('PC1')
plt.ylabel('PC2')

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())

plt.show()
