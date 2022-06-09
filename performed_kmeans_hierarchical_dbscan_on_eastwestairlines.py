"""
Created on Sun May 15 14:49:32 2022

@author: Sai pranay
"""

import pandas as pd

ewa = pd.ExcelFile("E:\\DATA_SCIENCE_ASS\\CLUSTERING\\EastWestAirlines.xlsx")
print(ewa)
ewa.sheet_names
ewa1 = pd.read_excel(ewa,sheet_name="data")
print(ewa1)
ewa1.head()
ewa1.shape
ewa1.info()
ewa1.describe()
ewa1.dtypes
ewa2 = ewa1.drop(['ID#'],axis=1)
ewa2



from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_s = scaler.fit_transform(ewa2)
x_s1 = pd.DataFrame(x_s)
x_s1

x = x_s1.iloc[:,0:11].values
x.shape
x


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

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

#=============

%matplotlib qt
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (16, 9)

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)
ax.plot(x[:,0],x[:,1],x[:,2],x[:,3],x[:,4],x[:,5],x[:,6],x[:,7],x[:,8],x[:,9],x[:,10])
plt.show()

# Initializing KMeans

from sklearn.cluster import KMeans
KMeans()
kmeans = KMeans(n_clusters=3)
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
ax.plot(x[:, 0], x[:, 1], x[:, 2],x[:,3],x[:,4],x[:,5],x[:,6],x[:,7],x[:,8],x[:,9],x[:,10])
ax.plot(C[:, 0], C[:, 1], C[:, 2],C[:,3],C[:,4],C[:,5],C[:,6],C[:,7],C[:,8],C[:,9],C[:,10], marker='*', c='Red', s=1000) # S is star size, c= * color

#########################################################

Y = pd.DataFrame(labels)

df_new = pd.concat([pd.DataFrame(x),Y],axis=1)
df_new

pd.crosstab(Y[0],Y[0])

Y


#-------------------------------------------------------------------------------
#==========================hireeachical==============================================

import scipy.cluster.hierarchy as shc

# construction of Dendogram

plt.figure(figsize=(10, 7))  
plt.title(" Dendograms")  
dend = shc.dendrogram(shc.linkage(x, method='complete'))


## Forming a group using clusters
from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='complete')
O = cluster.fit_predict(x)
O

plt.figure(figsize=(10, 7))  
plt.scatter(x[:,0], x[:,1], c=O, cmap='rainbow')  

Y_clust = pd.DataFrame(O)
Y_clust[0].value_counts()
##############################################################################

##  Implementing K-Means Clustering in Python ###



Y = pd.DataFrame(Y_clust)


df_new = pd.concat([pd.DataFrame(x),Y],axis=1)

pd.crosstab(Y[0],Y[0])

Y



#------------------------------------------------------------------------------
#----------------------------------DBSCAN--------------------------------------



from sklearn.cluster import DBSCAN
DBSCAN()
dbscan = DBSCAN(eps=3.7, min_samples=5)
dbscan.fit(x)

#Noisy samples are given the label -1.
dbscan.labels_

cl=pd.DataFrame(dbscan.labels_,columns=['cluster'])
cl
cl['cluster'].value_counts()


clustered = pd.concat([pd.DataFrame(x),cl],axis=1)
clustered

noisedata = clustered[clustered['cluster']==-1]
noisedata


finaldata = clustered[clustered['cluster']==0]
finaldata


a=0
while a<5:
  print(a)
  a=a+1


clustered.mean()
finaldata.mean()
