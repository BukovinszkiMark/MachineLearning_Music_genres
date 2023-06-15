# -*- coding: utf-8 -*-
"""
Created on Wed May 14 00:15:15 2021

@author: Márk Bukovinszki
"""

import numpy as np;
from urllib.request import urlopen;
from sklearn.decomposition import PCA;
from sklearn.feature_selection import SelectKBest;
import matplotlib.pyplot as plt;
from sklearn import model_selection as ms;
from sklearn import linear_model as lm;
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, roc_curve, auc, plot_roc_curve;
from sklearn import naive_bayes as nb;
from sklearn.cluster import KMeans;
from sklearn.metrics import davies_bouldin_score;


#--------------Reading the data-------------------

# Reading the dataset from url
# This dataset contains various statistics regarding music genres on Spotify 
# Original source: https://www.kaggle.com/yamaerenay/spotify-dataset-19212020-160k-tracks?select=data_by_genres_o.csv

url = "https://raw.githubusercontent.com/BukovinszkiMark/MachineLearning/main/data_by_genres_o.csv";
raw_data = urlopen(url);
data = np.loadtxt(raw_data, skiprows=1, delimiter=',', usecols=(2,3,5,6,7,8,9,10,11))
feature_names = ["acousticness","danceability","energy","instrumentalness","liveness","loudness","speechiness","tempo","valence",]

raw_data = urlopen(url);
target = np.loadtxt(raw_data, skiprows=1, delimiter=',', usecols=(12)) #Popularity


#--------------PCA-------------------

# Full PCA using scikit-learn
pca = PCA();
pca.fit(data);

# Visualizing the variance ratio which measures the importance of PCs
fig = plt.figure(1);
plt.title('Explained variance ratio plot');
var_ratio = pca.explained_variance_ratio_;
x_pos = np.arange(len(var_ratio));
plt.xticks(x_pos,x_pos+1);
plt.xlabel('Principal Components');
plt.ylabel('Variance');
plt.bar(x_pos,var_ratio, align='center', alpha=0.5);
plt.show(); 

#--------------SelectKBest-------------------

# Scatterplot of the first two most important features
feature_selection = SelectKBest(k=2);
feature_selection.fit(data, target);
scores = feature_selection.scores_;
features = feature_selection.transform(data);
mask = feature_selection.get_support();
feature_indices = [];
n = 9 #number of attributes
for i in range(n):
    if mask[i] == True : feature_indices.append(i);
x_axis, y_axis = feature_indices;
fig = plt.figure(2);
plt.title('Scatterplot for Popularity');
plt.xlabel(feature_names[x_axis]);
plt.ylabel(feature_names[y_axis]);
plt.scatter(data[:,x_axis],data[:,y_axis],s=50,c=target,cmap='gray_r');
plt.show();   

#--------------Another Scatterplot-------------------

# A manualy choosen Scatterplot
x_axis = 5 #Loudness
y_axis = 7 #Tempo
color = data[:,2] #Energy
fig = plt.figure(3);
plt.title('Scatterplot for Energy');
plt.xlabel(feature_names[x_axis]);
plt.ylabel(feature_names[y_axis]);
colors = ['blue','red','green'];
plt.scatter(data[:,x_axis],data[:,y_axis],s=50,c=color,cmap='gray_r');
plt.show();   

#--------------Predictions-------------------

# New Target for Logistic Regression
#   Creates new logistic value, that represents whether the genre 
#   is in the top 500 most popular
limit = sorted(target)[-500]
target_log = []
for i in range(len(target)):
    if target[i]>=limit:
        target_log.append(1)
    else:
        target_log.append(0)


# Partitioning into training and test sets
X_train, X_test, y_train, y_test = ms.train_test_split(data,target_log, test_size=0.3, 
                                shuffle = True);

# Fitting logistic regression
logreg_classifier = lm.LogisticRegression();
logreg_classifier.fit(X_train,y_train);
ypred_logreg = logreg_classifier.predict(X_train);   # spam prediction for train
cm_logreg_train = confusion_matrix(y_train, ypred_logreg); # train confusion matrix
cm_logreg_train_percent = confusion_matrix(y_train, ypred_logreg, normalize="pred"); # train confusion matrix
ypred_logreg = logreg_classifier.predict(X_test);   # spam prediction for test
cm_logreg_test = confusion_matrix(y_test, ypred_logreg); # test confusion matrix
cm_logreg_test_percent = confusion_matrix(y_test, ypred_logreg, normalize="pred"); # test confusion matrix
yprobab_logreg = logreg_classifier.predict_proba(X_test);  #  prediction probabilities

# Fitting naive Bayes classifier
naive_bayes_classifier = nb.GaussianNB();
naive_bayes_classifier.fit(X_train,y_train);
ypred_naive_bayes = naive_bayes_classifier.predict(X_train);  # spam prediction for train
cm_naive_bayes_train = confusion_matrix(y_train, ypred_naive_bayes); # train confusion matrix
cm_naive_bayes_train_percent = confusion_matrix(y_train, ypred_naive_bayes, normalize="pred"); # train confusion matrix
ypred_naive_bayes = naive_bayes_classifier.predict(X_test);  # spam prediction
cm_naive_bayes_test = confusion_matrix(y_test, ypred_naive_bayes); # test confusion matrix 
cm_naive_bayes_test_percent = confusion_matrix(y_test, ypred_naive_bayes, normalize="pred"); # test confusion matrix 
yprobab_naive_bayes = naive_bayes_classifier.predict_proba(X_test);  #  prediction probabilities

#--------------Evaluation of predictions-------------------

# Plot confusion matrixes
target_names = ["Not in TOP 500","In TOP 500"]

# Logreg
plot_confusion_matrix(logreg_classifier, X_test, y_test, display_labels=target_names, cmap="Blues")

# Bayes
plot_confusion_matrix(naive_bayes_classifier, X_test, y_test, display_labels=target_names, cmap="Blues")

# Logreg normalized
plot_confusion_matrix(logreg_classifier, X_test, y_test, display_labels=target_names, normalize="pred", cmap="Blues")

# Bayes normalized
plot_confusion_matrix(naive_bayes_classifier, X_test, y_test, display_labels=target_names, normalize="pred", cmap="Blues")



# Plotting ROC curves
plot_roc_curve(logreg_classifier, X_test, y_test);
plot_roc_curve(naive_bayes_classifier, X_test, y_test);

fpr_logreg, tpr_logreg, _ = roc_curve(y_test, yprobab_logreg[:,1]);
roc_auc_logreg = auc(fpr_logreg, tpr_logreg);

fpr_naive_bayes, tpr_naive_bayes, _ = roc_curve(y_test, yprobab_naive_bayes[:,1]);
roc_auc_naive_bayes = auc(fpr_naive_bayes, tpr_naive_bayes);

plt.figure(4);
lw = 2;
plt.plot(fpr_logreg, tpr_logreg, color='red',
         lw=lw, label='Logistic regression (AUC = %0.2f)' % roc_auc_logreg);
plt.plot(fpr_naive_bayes, tpr_naive_bayes, color='blue',
         lw=lw, label='Naive Bayes (AUC = %0.2f)' % roc_auc_naive_bayes);
plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--');
plt.xlim([0.0, 1.0]);
plt.ylim([0.0, 1.05]);
plt.xlabel('False Positive Rate');
plt.ylabel('True Positive Rate');
plt.title('Receiver operating characteristic curve');
plt.legend(loc="lower right");
plt.show();

#--------------Clustering-------------------

# Finding optimal cluster number
Max_K = 20;  # maximum cluster number
SSE = np.zeros((Max_K-2));  #  array for sum of squares errors
DB = np.zeros((Max_K-2));  # array for Davies Bouldin indeces
for i in range(Max_K-2):
    n_c = i+2
    kmeans = KMeans(n_clusters=n_c, random_state=2020);
    kmeans.fit(data);
    labels = kmeans.labels_;
    SSE[i] = kmeans.inertia_;
    DB[i] = davies_bouldin_score(data,labels);
    print("Clusters:", n_c, "SSE:", SSE[i], "DB:", DB[i])
    
    
# Visualization of SSE values    
fig = plt.figure(5);
plt.title('Sum of squares of error curve');
plt.xlabel('Number of clusters');
plt.ylabel('SSE');
plt.plot(np.arange(2,Max_K),SSE, color='red')
plt.show();

# Visualization of DB scores
fig = plt.figure(6);
plt.title('Davies-Bouldin score curve');
plt.xlabel('Number of clusters');
plt.ylabel('DB index');
plt.plot(np.arange(2,Max_K),DB, color='blue')
plt.show();    

# Default parameters
n_c = 3; # number of clusters

# Kmeans clustering
kmeans = KMeans(n_clusters=n_c, random_state=2020);  # instance of KMeans class
kmeans.fit(data);   #  fitting the model to data
labels = kmeans.labels_;  # cluster labels
centers = kmeans.cluster_centers_;  # centroid of clusters
sse = kmeans.inertia_;  # sum of squares of error (within sum of squares)
score = kmeans.score(data);  # negative error
# both sse and score measure the goodness of clustering

# Davies-Bouldin goodness-of-fit
DB = davies_bouldin_score(data,labels);

# PCA with limited components
pca = PCA(n_components=2);
pca.fit(data);
pc = pca.transform(data);  #  data coordinates in the PC space
centers_pc = pca.transform(centers);  # the cluster centroids in the PC space

# Visualizing of clustering in the principal components space
fig = plt.figure(7);
plt.title('Clustering of the data after PCA');
plt.xlabel('PC1');
plt.ylabel('PC2');
plt.scatter(pc[:,0],pc[:,1],s=50,c=labels);  # data
plt.scatter(centers_pc[:,0],centers_pc[:,1],s=200,c='red',marker='X');  # centroids
plt.legend();
plt.show();