# -*- coding: utf-8 -*-
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB 
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import warnings
import matplotlib.pyplot as plt
import numpy as np
import gnb, ttest

warnings.filterwarnings("ignore")

def loadDataset(dName):
    #print(dName)
    if(dName=='iris'):
        ds = datasets.load_iris()
        X,y = ds.data, ds.target
    elif(dName=='digits'):
        ds = datasets.load_digits()
        X,y = ds.data, ds.target
    elif(dName=='wine'):
        ds = datasets.load_wine()
        X,y = ds.data, ds.target
    elif(dName=='breast_cancer'):
        ds = datasets.load_breast_cancer()
        X,y = ds.data, ds.target
    elif(dName=='diabetes'):
        ds = datasets.load_diabetes()
        X,y = ds.data, ds.target
    elif(dName=='linnerud'):
        ds = datasets.load_linnerud()
        X,y = ds.target, ds.data
    elif(dName=='boston'):
        ds = datasets.load_boston()
        X,y = ds.data, ds.target
    return X,y

def linearReg(data_list, y_list, dName):
    if dName=='linnerud':
        y_list = y_list[:, np.newaxis, 0] #use chinup as feature
    X_train, X_test, y_train, y_test=train_test_split(data_list, y_list, test_size=0.30, random_state=42)
    model=LinearRegression()
    model.fit(data_list,y_list)
    y_pred = model.predict(X_test)
    print("Linear Regression Mean squared error: %.3f"
          % mean_squared_error(y_test, y_pred))
    mse = mean_squared_error(y_test, y_pred)
    return mse

def svmReg(data_list, y_list, dName):
    if dName=='linnerud':
        #data_list=data_list[:, np.newaxis, 0] #use chinup as feature
        y_list = y_list[:, np.newaxis, 0]
    #print(y_list)
    #print(data_list)
    X_train, X_test, y_train, y_test=train_test_split(data_list, y_list, test_size=0.30, random_state=42)
    model=svm.SVR()
    model.fit(data_list,y_list)
    y_pred = model.predict(X_test)
    print("SVM Mean squared error: %.3f"
          % mean_squared_error(y_test, y_pred))
    print("\n")
    mse = mean_squared_error(y_test, y_pred)
    return mse
    

#1(a)
########################################################################################################################################################
########################################################################################################################################################
finalData = []
names = []
datasetsArr = ['iris','digits','wine','breast_cancer']
for dsName in datasetsArr:
    X,y = loadDataset(dsName)
    #print(X,y)
    models = [('NB', GaussianNB()), ('Tree', DecisionTreeClassifier()), ('KNN', KNeighborsClassifier(n_neighbors=2)), ('MLP', MLPClassifier())]
    results = []
    for name, model in models:
        #print(model)
        cl = model.fit(X, y)
        tl = cl.predict(X)
        results.append(accuracy_score(tl, y))
        #print(dsName)
        names.append(name)
        #print(dsName)
    finalData.append(results)
print("\n")    
print("Q.1A")
print("###########################################################################################################")
print("4x4 matrix (dataset rows, classifier columns): ")
npClassArr = np.asarray(finalData, dtype=np.float32)
print(npClassArr)
fig, ax = plt.subplots(1,1,figsize=(10,10))
#plt.fig(figsize=(4,4))
fig.suptitle('Accuracy of 4 classifiers')
img = ax.imshow(finalData, extent=[0,1,0,1])
ax.set_xticks([0.125,0.375,0.625,0.875])
ax.set_yticks([0.125,0.375,0.625,0.875])
x_label_list = ['gnb', 'tree', 'knn', 'mlp']
y_label_list = ['breast cancer','wine','digits','iris']
ax.set_xticklabels(x_label_list)
ax.set_yticklabels(y_label_list)
plt.xlabel('classifiers')
plt.ylabel('datasets')
plt.tight_layout()
plt.colorbar(img)
plt.show()
print("Tree classifier has highest mean accuracy of "+str(np.mean(npClassArr[:,1]))+" across all datasets")
print("Iris dataset has the highest mean accuracy of "+str(npClassArr.mean(axis=1)[0])+" across all classifiers")
print("###########################################################################################################")


#1(b)
########################################################################################################################################################
########################################################################################################################################################
print("\n")    
print("Q.1B")
print("###########################################################################################################")
print("3x2 matrix (dataset rows, classifier columns): \n")
datasetsArr2 = ['boston','diabetes','linnerud']
fPerf = []
for dsName2 in datasetsArr2:
    data_list,y_list = loadDataset(dsName2)
    print(dsName2+" dataset: ")
    reg = ['linear','svm']
    mse=[]
    for rName in reg:
        if rName=='linear':
            mse.append(linearReg(data_list, y_list, dsName2))
        elif rName=='svm':
            mse.append(svmReg(data_list, y_list, dsName2))
    #print('Coefficients: \n', model.coef_)
    fPerf.append(mse)
npRegArr = np.asarray(fPerf, dtype=np.float32)
print(npRegArr)
fig1, ax1 = plt.subplots(1,1,figsize=(8,8))
#plt.fig(figsize=(4,4))
img1 = ax1.imshow(fPerf, cmap=plt.get_cmap('YlGnBu'))
plt.title('Performance of 2 regressions')
ax1.set_xticks([0,1])
ax1.set_yticks([0,1,2])
x_label_list2 = ['linear regression', 'support vector regression']
y_label_list2 = ['boston','diabetes','linnerud']
ax1.set_xticklabels(x_label_list2)
ax1.set_yticklabels(y_label_list2)
#plt.style.use('classic')
plt.xlabel('regressions')
plt.ylabel('datasets')
plt.tight_layout()
plt.colorbar(img1)
plt.show()
print("Linear regression technique has lower mean-squared error of "+str(np.amin(npRegArr.mean(axis=0)))+" across all datasets")
print("Linnerud dataset has lowest mean squared-error of "+str(np.amin(npRegArr.mean(axis=1)))+" across all regression methods")
print("###########################################################################################################")
print("\n")    

#2(a)
########################################################################################################################################################
########################################################################################################################################################
gnb.gnb()

#3(a)
##############################################################################################
##############################################################################################
ttest.plotAvgROC()
