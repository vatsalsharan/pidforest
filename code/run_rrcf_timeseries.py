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
import rrcf

datasets = ['nyc_taxi','ambient_temperature_system_failure','cpu_utilization_asg_misconfiguration','machine_temperature_system_failure']
L = len(datasets)
trials = 5
run_lof_svm = 0

for i in range(0,L):
    data = pd.read_csv('../data/numenta/'+datasets[i]+'.csv')
    value = list(data["value"])
    arr = np.array(value)
    y = np.array(list(data["label"]))
    X = ts.shingle(arr, 10)
    X = np.transpose(X)
    t1, _ = np.shape(X)
    [n,d] = np.shape(X)
    y = y[:t1]
    file_name = 'experiment_results/' + datasets[i] + '_rrcf.txt'
    File_object = open(file_name,"w")   
    time_all = np.zeros((trials,4))
    precision_all = np.zeros((trials,4))
    auc_all = np.zeros((trials,4))
    
    for j in range(0,trials):
    
        print('\n\n******'+datasets[i]+' trial '+str(j+1)+'*******\n\n')
        
        print('\n******RRCF*******\n')
        num_trees = 500
        tree_size = 256
        start = time.time()
        forest = []
        while len(forest) < num_trees:
            # Select random subsets of points uniformly from point set
            ixs = np.random.choice(n, size=(n // tree_size, tree_size),
                   replace=False)
            # Add sampled trees to forest
            trees = [rrcf.RCTree(X[ix], index_labels=ix) for ix in ixs]
            forest.extend(trees)
        
        end = time.time()
        time_all[j,0] = end - start
        
        # Compute average CoDisp
        avg_codisp = pd.Series(0.0, index=np.arange(n))
        index = np.zeros(n)
        for tree in forest:
            codisp = pd.Series({leaf : tree.codisp(leaf) for leaf in tree.leaves})
            avg_codisp[codisp.index] += codisp
            np.add.at(index, codisp.index.values, 1)
        avg_codisp /= index
        iso_scores = avg_codisp
            
        
        
        precision_iso, recall_iso, thresholds_iso = metrics.precision_recall_curve(y, iso_scores, pos_label=1)
        
        precision_all[j,0] = max(2*precision_iso*recall_iso/(precision_iso+recall_iso))
        
        
        auc_all[j,0] = metrics.roc_auc_score(y, iso_scores)
        
        
        for k in range(0,1):
            print('{:.4f}\t'.format( precision_all[j,k] ))
        print('\n')
        
        for k in range(0,1):
            print('{:.4f}\t'.format( auc_all[j,k] ))
        print('\n')
                
    File_object.write('\n\nRRCF\n\n')    
        
    for j in range(0,trials):
        for k in range(0,1):
            File_object.write('{:.4f}\t'.format( precision_all[j,k] ))
        File_object.write('\n')
        
    File_object.write('\n')   
        
    for k in range(0,1):
            File_object.write('{:.4f}\t'.format( np.mean(precision_all[:,k]) ))
    File_object.write('\n')   
    
    for k in range(0,1):
            File_object.write('{:.4f}\t'.format( np.std(precision_all[:,k]) ))
    File_object.write('\n')
    
    File_object.write('\n\nRRCF\n\n')    
        
    for j in range(0,trials):
        for k in range(0,1):
            File_object.write('{:.4f}\t'.format( auc_all[j,k] ))
        File_object.write('\n')
        
    File_object.write('\n') 
        
    for k in range(0,1):
            File_object.write('{:.4f}\t'.format( np.mean(auc_all[:,k]) ))
    File_object.write('\n')    
    
    for k in range(0,1):
            File_object.write('{:.4f}\t'.format( np.std(auc_all[:,k]) ))
    File_object.write('\n')
        
    File_object.close()
    
    file_name = 'experiment_results/' + datasets[i] + '_rrcf_results.mat'
    sio.savemat(file_name, {'time_all':time_all, 'precision_all':precision_all, 'auc_all':auc_all})
    