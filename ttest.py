# -*- coding: utf-8 -*-
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import fetch_california_housing
from scipy import interp
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

def checkIfSignificant(val):
    txt = ''
    if val<0.05:
        txt = 'is significant'
    return txt

def knn(X_data,Y_data,cv):
        knn = KNeighborsClassifier()
        plt.figure(figsize=(15,15))
        tprs = []
        aucs = []
        acc = []
        mean_fpr = np.linspace(0, 1, 100)
        i=0
        for train, test in cv.split(X_data):
            correct = 0
            knn.fit(X_data[train],Y_data[train])
            probab = knn.predict_proba(X_data[test])
            y_pred = knn.predict(X_data[test])
            correct += (y_pred == Y_data[test]).sum().item()
            acc.append(correct/len(y_pred))
            fpr,tpr,thresholds = roc_curve(Y_data[test],probab[:,1])
            aucc  = auc(fpr,tpr)
            aucs.append(aucc)
            tprs.append(interp(mean_fpr,fpr,tpr))
            labelTxt = 'ROC fold '+str(i+1)+' (AUC = '+str(aucc)+')'
            plt.plot(fpr,tpr,lw=1,alpha = 0.3,label=labelTxt)
            i=i+1
        mean_tpr = np.mean(tprs,axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr,mean_tpr)
        std_auc = np.std(aucs)
        plt.errorbar(mean_fpr,mean_tpr,xerr=std_auc,yerr=std_auc,alpha=.9,label='Standard Deviation: '+str(std_auc))
        plt.legend()
        plt.title("Mean Area under Curve for Knn : "+str(mean_auc))
        plt.show()
        return (mean_auc,acc)
    
def gnb(X_data,Y_data,cv):
        plt.figure(figsize=(15,15))
        gnb = GaussianNB()
        tprs = []
        aucs = []
        acc = []
        mean_fpr = np.linspace(0, 1, 100)
        i=0
        for train, test in cv.split(X_data):
            correct = 0
            gnb.fit(X_data[train],Y_data[train])
            probab = gnb.predict_proba(X_data[test])
            y_pred = gnb.predict(X_data[test])
            correct += (y_pred == Y_data[test]).sum().item()
            acc.append(correct/len(y_pred))
            fpr,tpr,thresholds = roc_curve(Y_data[test],probab[:,1])
            aucc  = auc(fpr,tpr)
            aucs.append(aucc)
            tprs.append(interp(mean_fpr,fpr,tpr))
            labelTxt = 'ROC fold '+str(i+1)+' (AUC = '+str(aucc)+')'
            plt.plot(fpr,tpr,lw=1,alpha = 0.3,label=labelTxt)
            i=i+1
        mean_tpr = np.mean(tprs,axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr,mean_tpr)
        std_auc = np.std(aucs)
        plt.errorbar(mean_fpr,mean_tpr,xerr=std_auc,yerr=std_auc,alpha=.9,label='Standard Deviation: '+str(std_auc))
        plt.legend()
        plt.title("Mean Area under Curve for GNB : "+str(mean_auc))
        plt.show()
        return (mean_auc,acc)
        
def randomTree(X_data,Y_data,cv):
    plt.figure(figsize=(15,15))
    rtree = RandomForestClassifier()
    tprs = []
    aucs = []
    acc = []
    mean_fpr = np.linspace(0, 1, 100)
    i=0
    for train, test in cv.split(X_data):
        correct = 0
        rtree.fit(X_data[train],Y_data[train])
        probab = rtree.predict_proba(X_data[test])
        y_pred = rtree.predict(X_data[test])
        correct += (y_pred == Y_data[test]).sum().item()
        acc.append(correct/len(y_pred))
        fpr,tpr,thresholds = roc_curve(Y_data[test],probab[:,1])
        aucc  = auc(fpr,tpr)
        aucs.append(aucc)
        tprs.append(interp(mean_fpr,fpr,tpr))
        labelTxt = 'ROC fold '+str(i+1)+' (AUC = '+str(aucc)+')'
        plt.plot(fpr,tpr,lw=1,alpha = 0.3,label=labelTxt)
        i=i+1
    mean_tpr = np.mean(tprs,axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr,mean_tpr)
    std_auc = np.std(aucs)
    plt.errorbar(mean_fpr,mean_tpr,xerr=std_auc,yerr=std_auc,alpha=.9,label='Standard Deviation: '+str(std_auc))
    plt.legend()
    plt.title("Mean Area under Curve for Random Forest : "+str(mean_auc))
    plt.show()
    return (mean_auc,acc)

def t_test(a,b):
    var_a = np.asarray(a).var(ddof=1)
    var_b = np.asarray(b).var(ddof=1)
    s = np.sqrt((var_a+var_b)/2)
    N = a.size
    t = (a.mean()-b.mean())/(s*np.sqrt(2/N))
    df = 2*N-2
    p = 1-stats.t.cdf(t,df=df)
    return(2*p)


def plotAvgROC():
    print("Q.3A")
    print("###########################################################################################################")

    cali = fetch_california_housing()
    x_data = cali.data
    y_data = cali.target
    y_data = np.where(y_data>3.0,1,0)
    cv = KFold(n_splits = 10,random_state=None,shuffle=False)
    
    #gnb
    GNB_auc, GNB_acc = gnb(x_data,y_data,cv)
    GNB_acc_mean = np.mean(GNB_acc)
    print("The area under curve for GNB Classifier is: ",GNB_auc)
    print("Performance of GNB: ", GNB_acc_mean)
    
    #random forest
    Tree_auc, Tree_acc = randomTree(x_data,y_data,cv)
    Tree_acc_mean = np.mean(Tree_acc)
    print("The area under curve for Random Forest Classifier is: ",Tree_auc)
    print("Performance of Random Forest: ",Tree_acc_mean)
    
    ##knn 
    knn_auc,knn_acc = knn(x_data,y_data,cv)
    knn_acc_mean = np.mean(knn_acc)
    print("The area under curve for KNN is: ",knn_auc)
    print("Performance of KNN: ",knn_acc_mean)
    
    print("\n")
    print("Comparisons:")
    if GNB_auc>knn_auc and GNB_auc>Tree_auc:
        print("GNB has highest area under curve with area: " , GNB_auc)
    elif Tree_auc>knn_auc and GNB_auc<Tree_auc:
        print("Random Forest has highest area under curve with area: " , Tree_auc)
    elif Tree_auc<knn_auc and GNB_auc<knn_auc:
        print("KNN has highest area under curve with area: " , knn_auc)
    
    #T-test and P-value
    if GNB_acc_mean>knn_acc_mean and GNB_acc_mean>Tree_acc_mean:
        print("GNB has has highest performance with accuracy: " , GNB_acc_mean)
        print("\n")
        print("T-test and P-values:")
        p1 = t_test(np.asarray(GNB_acc),np.asarray(knn_acc))
        cSig = checkIfSignificant(p1)
        print("When we compare Gausian Naive Bayes and Knn, P-value: "+str(p1)+' '+cSig)
        p2 = t_test(np.asarray(GNB_acc),np.asarray(Tree_acc))
        cSig = checkIfSignificant(p2)
        print("When we compare Random Forest and Gausian Naive Bayes, P-value: "+str(p2)+' '+cSig)
    elif Tree_acc_mean>knn_acc_mean and GNB_acc_mean<Tree_acc_mean:
        print("Random Forest has highest performance with accuracy: " , Tree_acc_mean )
        print("\n")
        print("T-test and P-values:")
        p1 = t_test(np.asarray(Tree_acc),np.asarray(knn_acc))
        cSig = checkIfSignificant(p1)
        print("When we compare Random Forest and Knn, P-value: "+str(p1)+' '+cSig)
        p2 = t_test(np.asarray(Tree_acc),np.asarray(GNB_acc))
        cSig = checkIfSignificant(p2)
        print("When we compare Random Forest and Gausian Naive Bayes, P-value: "+str(p2)+' '+cSig)
    elif Tree_acc_mean<knn_acc_mean and GNB_acc_mean<knn_acc_mean:
        print("KNN has highest performance with accuracy: " , knn_acc_mean)
        print("\n")
        print("T-test and P-values:")
        p1 = t_test(np.asarray(knn_acc),np.asarray(Tree_acc))
        cSig = checkIfSignificant(p1)
        print("When Random Forest and Knn are compared, P-value: "+str(p1)+' '+cSig)
        p2 = t_test(np.asarray(knn_acc),np.asarray(GNB_acc))
        cSig = checkIfSignificant(p2)
        print("When knn and Gausian Naive Bayes are compared, P-value: "+str(p2)+' '+cSig)

    print("###########################################################################################################")
