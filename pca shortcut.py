import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from matplotlib import*

from sklearn.decomposition import PCA 


data = pd.read_csv('gene_data-.csv')
meta = pd.read_csv('Meta-data-sheet.csv')


len(data)

dta_ch = data.iloc[:,2:]
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

z = pca.explained_variance_
a = 0.0
for i in range(len(z)):
    a = a+ float(z[i])
print (a)

xn = pca.fit_transform(X)

xn.shape

target = np.array(meta['Time'])

x_min, x_max = xn[:, 0].min() - 10000, xn[:, 0].max() + 10000
y_min, y_max = xn[:, 1].min() - 10000, xn[:, 1].max() + 10000

plt.figure(figsize= (15,10))

plt.scatter(xn[:, 0], xn[:, 1], c=target, cmap=plt.cm.Set1,
            edgecolor='k')
plt.xlabel('PC1')
plt.ylabel('PC2')

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())

plt.show()
