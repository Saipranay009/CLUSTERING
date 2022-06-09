# -*- coding: utf-8 -*-
"""
Created on Mon May  2 18:47:33 2022

@author: Sai pranay
"""
#-------------------------------imortinf the data set--------------------------

import pandas as pd

cd = pd.read_csv("E:\DATA_SCIENCE_ASS\CLUSTERING\\crime_data.csv")
list(cd)
cd.head()
cd.shape
cd.info()
cd.describe()
cd.dtypes
cd.value_counts()

#-------dropping

cd1 = cd.drop(['Unnamed: 0'],axis=1)
cd1
#--------standardization

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scale = scaler.fit_transform(cd1)

X1 = pd.DataFrame(X_scale)
X1

x = X1.iloc[:,:].values
x
x.shape


##############################################################################
#----------------------K-MEAN CLUSTER
%matplotlib qt
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (16, 9)

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(x[:,0], x[:, 1], x[:, 2],x[:,3])
plt.show()

# Initializing KMeans
from sklearn.cluster import KMeans
KMeans()
kmeans = KMeans(n_clusters=4)
# Fitting with inputs
kmeans = kmeans.fit(x)
# Predicting the clusters
labels = kmeans.predict(x)
type(labels)
# Getting the cluster centers
C = kmeans.cluster_centers_
C
# Total with in centroid sum of squares 
kmeans.inertia_


%matplotlib qt
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(x[:, 0], x[:, 1], x[:, 2],x[:,3])
ax.scatter(C[:, 0], C[:, 1], C[:, 2],C[:,3], marker='*', c='Red', s=1000) # S is star size, c= * color



Y = pd.DataFrame(labels)


df_new = pd.concat([pd.DataFrame(x),Y],axis=1)

pd.crosstab(Y[0],Y[0])

print(Y)


clust = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i,random_state=0)
    kmeans.fit(x)
    clust.append(kmeans.inertia_)
    
plt.plot(range(1, 11), clust)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('inertial values')
plt.show()

print(clust)

#---------------------------------------------------------------------------------------------------
#--------------------------------Hierarchical CLUSTERS


import scipy.cluster.hierarchy as shc

# construction of Dendogram
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 7))  
plt.title("Customer Dendograms")  
dend = shc.dendrogram(shc.linkage(x, method='complete')) 

"""
Now we know the number of clusters for our dataset, 
the next step is to group the data points into these five clusters. 
To do so we will again use the AgglomerativeClustering
"""
## Forming a group using clusters
from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='complete')
O = cluster.fit_predict(x)
O


plt.figure(figsize=(10, 7))  
plt.scatter(x[:,0], x[:,1],x[:,2], c=O, cmap='rainbow')  

Y_clust = pd.DataFrame(O)
Y_clust[0].value_counts()



Y1 = pd.DataFrame(Y_clust)
Y1

df_new1 = pd.concat([pd.DataFrame(x),Y1],axis=1)

pd.crosstab(Y1[0],Y1[0])





#------------------------------------------------------------------------------
#---------------------------DBSCAN---------------------------------------------


from sklearn.cluster import DBSCAN
DBSCAN()
dbscan = DBSCAN(eps=2, min_samples=4)
dbscan.fit(x)

#Noisy samples are given the label -1.
dbscan.labels_

cl2=pd.DataFrame(dbscan.labels_,columns=['cluster1'])
cl2
cl2['cluster1'].value_counts()


XX = pd.DataFrame(x)
XX

clustered = pd.concat([XX,cl2],axis=1)
clustered


noisedata = clustered[clustered['cluster1']==-1]
noisedata

finaldata = clustered[clustered['cluster1']==0]
finaldata



a=0
while a<5:
  print(a)
  a=a+1


clustered.mean()
finaldata.mean()
