# -*- coding: utf-8 -*-
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.naive_bayes import GaussianNB 
from sklearn.model_selection import cross_val_score
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import FastICA

def pltRocCurve(y_true, y_score, title):
    auc = roc_auc_score(y_true, y_score)
    fpr, tpr, thresholds = roc_curve(y_true,y_score) 
    plt.subplots(1,1,figsize=(10,10))
    plt.plot([0,1],[0,1],linestyle = '--',lw = 2,color = 'black')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.plot(fpr, tpr, linestyle='--', label='ROC curve')
    plt.text(0.32,0.7,'More accurate area',fontsize = 12)
    plt.text(0.63,0.4,'Less accurate area',fontsize = 12)
    title = title+' with area under ROC curve: '+str(auc)
    plt.title(title)
    plt.legend()
    plt.show()
    print('The area under the '+title+' is: '+str(auc))
    
def calcGnb(caliData, caliTarget, pType):
    gnb = GaussianNB() 
    gnb.fit(caliData, caliTarget) 
    if pType==1:
        probRes = gnb.predict_proba(caliData)
    elif pType==2:
        probRes = gnb.predict(caliData)
    return probRes
    
def gnb():
   
    caliData = fetch_california_housing().data
    caliTarget = fetch_california_housing().target
    #X= np.column_stack((caliData,caliTarget))
    #print(fetch_california_housing().feature_names)
    
    price = 300000
    for i in range(0,20640):
        #print(caliTarget[i])
        tVal = caliTarget[i]
        #print(i)
        #print(X[i:i+1,8])
        tVal=tVal*100000
        #print(i)
        if tVal > price:
            #X[i:i+1,8]=1
            caliTarget[i] = 1
        elif tVal <= price:
            #X[i:i+1,8]=0
            caliTarget[i] = 0


    #X_train, X_test, y_train, y_test = train_test_split(caliData, caliTarget, test_size=0.4, random_state=1) 
    '''gnb = GaussianNB() 
    gnb.fit(caliData, caliTarget) 
    gnbProb = gnb.predict_proba(caliData)
    ResubTrainData = gnb.predict(caliData)'''
    gnbProb = calcGnb(caliData, caliTarget, 1)
    ResubTrainData = calcGnb(caliData, caliTarget, 2)
    print("Q.2A")
    print("###########################################################################################################")
    print("Probability that the house is worth below $300k:\n",gnbProb[:,0],"\n")
    print("Probability that the house is worth over $300k:\n",gnbProb[:,1])
    print("###########################################################################################################\n")
    print("\n")    

#2(b)
##############################################################################################
##############################################################################################
    scores = cross_val_score(GaussianNB(), caliData, caliTarget, cv=10)
    correct = 0
    #print(scores)
    #print(np.mean(scores))
    print("Q.2B")
    print("###########################################################################################################")
    avgError = 1-np.mean(scores)
    print("The average error across all folds on the test set: ",avgError,"\n")
    #print(int(gnbProb.sum()))
    correct += (ResubTrainData==caliTarget).sum().item()
    resubErr = 1-correct/gnbProb.sum()
    print("The resubstitution error (error on training data): ",resubErr,"\n")
    if avgError > resubErr:
        errDiff = avgError - resubErr
        print("Considering above results it is clear that the average error across all folds on the test set is greater than the resubstitution error (error on training data). Hence we can say that the resubstitution error rate is less.")
    else:
        errDiff = resubErr - avgError
        print("Considering above results it is clear that the average error across all folds on the test set is smaller than the resubstitution error (error on training data). Hence we can say that the average error rate is less.")
        
    print("Though the error difference is "+str(errDiff)+" which is very minimal.")
    print("###########################################################################################################\n")
    print("\n")    

#2(c)
##############################################################################################
##############################################################################################
    print("Q.2C")
    print("###########################################################################################################")
    pltRocCurve(caliTarget, gnbProb[:,1], 'ROC curve using the training data from 2a')
    sevenFeature = np.delete(caliData,[2, 3], 1)
    gnbProbSf = calcGnb(sevenFeature, caliTarget, 1)
    #print(sevenFeature.shape)
    pltRocCurve(caliTarget, gnbProbSf[:,1], 'ROC curve after removing 2 features')
    #PCA transformation of training data
    pca = PCA(n_components=2)
    pca.fit(caliData, caliTarget) 
    pcaData = pca.transform(caliData)
    gnbProbPca = calcGnb(pcaData, caliTarget, 1)
    pltRocCurve(caliTarget, gnbProbPca[:,1], 'ROC curve after PCA transformation')
    #FASTICA transformation on training data
    fastica = FastICA(n_components=7, random_state=0)
    fasticadata = fastica.fit_transform(caliData)
    gnbProbPca = calcGnb(fasticadata, caliTarget, 1)
    pltRocCurve(caliTarget, gnbProbPca[:,1], 'ROC curve after FastICA transformation')
    #PLS Regression on training data
    pls2 = PLSRegression(n_components=2)
    pls2.fit(caliData, caliTarget)
    plsData = pls2.transform(caliData)
    gnbProbPls = calcGnb(plsData, caliTarget, 1)
    pltRocCurve(caliTarget, gnbProbPls[:,1], 'ROC curve after PLS transformation')
    #print(plsData)
    print("\nAbove results prove that we can increase area under curve by removing 2 features, performing FastICA transformation and performing PLS transformation. Performing PCA transformation decreases the area under ROC curve.")
    print("###########################################################################################################\n")
    print("\n")    