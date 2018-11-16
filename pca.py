# In[1]:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from matplotlib import*


# In[2]:
data = pd.read_csv('g.csv')        #here g.csv is same file as Assignment-gene_data; and m.csv is same file as Assignment-meta_data
meta = pd.read_csv('m.csv')


# In[3]:
dta_ch = data.iloc[:,2:]
dta_ch=dta_ch.replace({'ssssss': '52.0', 'hhhh' : '358.43' ,'NA' : '150.26'}, regex=True)


# In[4]:
len(data)

# In[5]:
X=dta_ch.head(22411)
X=X.astype(float)


X[np.isnan(X)] = np.median(X[~np.isnan(X)])


X = X.T
X.shape


X_std = StandardScaler().fit_transform(X)


mean_vector = np.mean(X_std, axis=0)
covariance_matrix = (X_std - mean_vector).T.dot((X_std - mean_vector)) / (X_std.shape[0]-1)


covariance_matrix = np.cov(X_std.T)


eig_values, eig_vectors = np.linalg.eig(covariance_matrix)


u,s,v = np.linalg.svd(X_std.T)
u


eig_pairs = [(np.abs(eig_values[i]), eig_vectors[:,i]) for i in range(len(eig_values))]
eig_pairs.sort(key=lambda x: x[0], reverse=True)
print('Eigenvalues from largest to smallest:')
for i in eig_pairs:
    print(i[0])
    
    
    import numpy as np
matrix_k = np.hstack((eig_pairs[0][1].reshape(22411,1),
                      eig_pairs[1][1].reshape(22411,1)))
                      
                      
total = sum(eig_values)
var_exp = [(i / total)*100 for i in sorted(eig_values, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

Y = X_std.dot(matrix_k)


sample = np.array(meta['Time'])


plt.figure(figsize= (6,4))

plt.scatter(Y[:, 0], Y[:, 1], c=sample, cmap=plt.cm.Set1,
            edgecolor='k')
plt.xlabel('PC1')
plt.ylabel('PC2')


plt.show()
# In[6]:
print(var_exp)
