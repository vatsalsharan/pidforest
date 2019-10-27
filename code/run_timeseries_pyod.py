import numpy as np
import matplotlib.pyplot as plt
from scripts.forest import Forest
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
import scripts.timeseries as ts
import pandas as pd
import time
import scipy.io as sio
from sklearn import metrics
from pyod.models.knn import KNN
from pyod.models.pca import PCA

datasets = ['nyc_taxi','ambient_temperature_system_failure','cpu_utilization_asg_misconfiguration','machine_temperature_system_failure']
L = len(datasets)
trials = 1
run_lof_svm = 1

for i in range(0,L):
    data = pd.read_csv('../data/numenta/'+datasets[i]+'.csv')
    value = list(data["value"])
    arr = np.array(value)
    amazon_scores = np.array(list(data["anomaly_score"]))
    y = np.array(list(data["label"]))
    X = ts.shingle(arr, 10)
    X = np.transpose(X)
    t1, _ = np.shape(X)
    y = y[:t1]
    file_name = 'experiment_results/' + datasets[i] + '_pyod.txt'
    File_object = open(file_name,"w")   
    time_all = np.zeros((trials,4))
    precision_all = np.zeros((trials,4))
    auc_all = np.zeros((trials,4))
    
    for j in range(0,trials):
    
        print('\n\n******'+datasets[i]+' trial '+str(j+1)+'*******\n\n')
        
        print('\n******kNN*******\n')
        start = time.time()
        clf = KNN()
        clf.fit(X)
        end = time.time()
        time_all[j,0] = end - start
        iso_scores = clf.decision_scores_ 
    
       
        
        print('\n******PCA*******\n')
        start = time.time()
        clf = PCA()
        clf.fit(X)
        end = time.time()
        time_all[j,1] = end - start
        lof_scores = clf.decision_scores_

        
        
        
        precision_iso, recall_iso, thresholds_iso = metrics.precision_recall_curve(y, iso_scores, pos_label=1)
        precision_lof, recall_lof, thresholds_lof = metrics.precision_recall_curve(y, lof_scores, pos_label=1)
        
        precision_all[j,0] = max(2*precision_iso*recall_iso/(precision_iso+recall_iso))
        precision_all[j,1] = max(2*precision_lof*recall_lof/(precision_lof+recall_lof))
        
        
        auc_all[j,0] = metrics.roc_auc_score(y, iso_scores)
        auc_all[j,1] = metrics.roc_auc_score(y, lof_scores)
       
        
        for k in range(0,2):
            print('{:.4f}\t'.format( precision_all[j,k] ))
        print('\n')
        
        for k in range(0,2):
            print('{:.4f}\t'.format( auc_all[j,k] ))
        print('\n')
                
    File_object.write('\n\nkNN\tPCA\n\n')    
        
    for j in range(0,trials):
        for k in range(0,2):
            File_object.write('{:.4f}\t'.format( precision_all[j,k] ))
        File_object.write('\n')
        
    File_object.write('\n')   
        
    for k in range(0,2):
            File_object.write('{:.4f}\t'.format( np.mean(precision_all[:,k]) ))
    File_object.write('\n')   
    
    for k in range(0,2):
            File_object.write('{:.4f}\t'.format( np.std(precision_all[:,k]) ))
    File_object.write('\n')
    
    File_object.write('\n\nkNN\tPCA\n\n')    
        
    for j in range(0,trials):
        for k in range(0,2):
            File_object.write('{:.4f}\t'.format( auc_all[j,k] ))
        File_object.write('\n')
        
    File_object.write('\n') 
        
    for k in range(0,2):
            File_object.write('{:.4f}\t'.format( np.mean(auc_all[:,k]) ))
    File_object.write('\n')    
    
    for k in range(0,2):
            File_object.write('{:.4f}\t'.format( np.std(auc_all[:,k]) ))
    File_object.write('\n')
        
    File_object.close()
    
    file_name = 'experiment_results/' + datasets[i] + '_results_pyod.mat'
    sio.savemat(file_name, {'time_all':time_all, 'precision_all':precision_all, 'auc_all':auc_all})

    