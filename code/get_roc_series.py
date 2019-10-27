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
trials = 1

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
    [n,d] = np.shape(X)

    auc_all = np.zeros((trials,7))
    
    for j in range(0,trials):
    
        print('\n\n******'+datasets[i]+' trial '+str(j+1)+'*******\n\n')
        f = plt.figure()
        plt.rcParams.update({'font.size': 14})
        
        print('\n******PIDForest*******\n')
        n_samples = 100
        kwargs = {'max_depth': 10, 'n_trees':50,  'max_samples': n_samples, 'max_buckets': 3, 'epsilon': 0.1, 'sample_axis': 1, 
          'threshold': 0}
        forest = Forest(**kwargs)
        forest.fit(np.transpose(X))
        indices, outliers, scores , pst, alg_scores = forest.predict(np.transpose(X), err = 0.1, pct=50)
        alg_scores = - alg_scores
        fpr_alg, tpr_alg, thresholds_alg = metrics.roc_curve(y, alg_scores, pos_label=1)
        #plt.plot(fpr_alg,tpr_alg, 'b')
        thresh_len = len(fpr_alg)
        sample_thresh = np.int_( [k * thresh_len/10 for k in range(10)] )
        sample_thresh = np.concatenate( [sample_thresh, np.asarray([thresh_len-1]) ])
        #plt.scatter(fpr_alg[sample_thresh],tpr_alg[sample_thresh], c = 'b', marker = ".", s=100, label='PIDForest')
        plt.plot(fpr_alg[sample_thresh],tpr_alg[sample_thresh], c = 'k', marker = ">", markersize=10, label='PIDForest')
        auc_all[j,0] = metrics.roc_auc_score(y, alg_scores)
        
        print('\n******iForest*******\n')
        clf = IsolationForest(contamination = 0.1, behaviour = 'new')
        clf.fit(X)
        alg_scores = clf.score_samples(X)
        alg_scores = - alg_scores
        fpr_alg, tpr_alg, thresholds_alg = metrics.roc_curve(y, alg_scores, pos_label=1)
        #plt.plot(fpr_alg,tpr_alg, 'g')
        thresh_len = len(fpr_alg)
        sample_thresh = np.int_( [k * thresh_len/10 for k in range(10)] )
        sample_thresh = np.concatenate( [sample_thresh, np.asarray([thresh_len-1]) ])
        #plt.scatter(fpr_alg[sample_thresh],tpr_alg[sample_thresh], c = 'g', marker = "v", s=100, label='iForest')
        plt.plot(fpr_alg[sample_thresh],tpr_alg[sample_thresh], c = 'g', marker = "v", markersize=10, label='iForest')
        auc_all[j,1] = metrics.roc_auc_score(y, alg_scores)
        
        print('\n******RRCF*******\n')
        num_trees = 500
        tree_size = 256
        forest = []
        while len(forest) < num_trees:
            # Select random subsets of points uniformly from point set
            ixs = np.random.choice(n, size=(n // tree_size, tree_size),
                   replace=False)
            # Add sampled trees to forest
            trees = [rrcf.RCTree(X[ix], index_labels=ix) for ix in ixs]
            forest.extend(trees)
                
        # Compute average CoDisp
        avg_codisp = pd.Series(0.0, index=np.arange(n))
        index = np.zeros(n)
        for tree in forest:
            codisp = pd.Series({leaf : tree.codisp(leaf) for leaf in tree.leaves})
            avg_codisp[codisp.index] += codisp
            np.add.at(index, codisp.index.values, 1)
        avg_codisp /= index
        alg_scores = avg_codisp
        fpr_alg, tpr_alg, thresholds_alg = metrics.roc_curve(y, alg_scores, pos_label=1)
        #plt.plot(fpr_alg,tpr_alg, 'r')
        thresh_len = len(fpr_alg)
        sample_thresh = np.int_( [k * thresh_len/10 for k in range(10)] )
        sample_thresh = np.concatenate( [sample_thresh, np.asarray([thresh_len-1]) ])
        #plt.scatter(fpr_alg[sample_thresh],tpr_alg[sample_thresh], c = 'r', marker = "*", s=100, label='RRCF')
        plt.plot(fpr_alg[sample_thresh],tpr_alg[sample_thresh], c = 'r', marker = "*", markersize=10, label='RRCF')
        auc_all[j,2] = metrics.roc_auc_score(y, alg_scores)
        
        if False:
        
            print('\n******LOF*******\n')
            clf = LocalOutlierFactor()
            clf.fit(X)
            alg_scores = clf.negative_outlier_factor_
            alg_scores = - alg_scores
            fpr_alg, tpr_alg, thresholds_alg = metrics.roc_curve(y, alg_scores, pos_label=1)
            #plt.plot(fpr_alg,tpr_alg, 'm')
            thresh_len = len(fpr_alg)
            sample_thresh = np.int_( [k * thresh_len/10 for k in range(10)] )
            sample_thresh = np.concatenate( [sample_thresh, np.asarray([thresh_len-1]) ])
            #plt.scatter(fpr_alg[sample_thresh],tpr_alg[sample_thresh], c = 'm', marker = "D", s=100, label='LOF')
            plt.plot(fpr_alg[sample_thresh],tpr_alg[sample_thresh], c = 'm', marker = "D", markersize=10, label='LOF')
            auc_all[j,3] = metrics.roc_auc_score(y, alg_scores)

            print('\n******SVM*******\n')
            clf = OneClassSVM(kernel='rbf')
            clf.fit(X)
            alg_scores = clf.score_samples(X)
            alg_scores = - alg_scores
            fpr_alg, tpr_alg, thresholds_alg = metrics.roc_curve(y, alg_scores, pos_label=1)
            #plt.plot(fpr_alg,tpr_alg, 'c')
            thresh_len = len(fpr_alg)
            sample_thresh = np.int_( [k * thresh_len/10 for k in range(10)] )
            sample_thresh = np.concatenate( [sample_thresh, np.asarray([thresh_len-1]) ])
            #plt.scatter(fpr_alg[sample_thresh],tpr_alg[sample_thresh], c = 'c', marker = "s", s=100, label='SVM')
            plt.plot(fpr_alg[sample_thresh],tpr_alg[sample_thresh], c = 'c', marker = "s", markersize=10, label='SVM')
            auc_all[j,4] = metrics.roc_auc_score(y, alg_scores)
        
        print('\n******kNN*******\n')
        clf = KNN()
        clf.fit(X)
        alg_scores = clf.decision_scores_ 
        fpr_alg, tpr_alg, thresholds_alg = metrics.roc_curve(y, alg_scores, pos_label=1)
        #plt.plot(fpr_alg,tpr_alg, 'y')
        thresh_len = len(fpr_alg)
        sample_thresh = np.int_( [k * thresh_len/10 for k in range(10)] )
        sample_thresh = np.concatenate( [sample_thresh, np.asarray([thresh_len-1]) ])
        #plt.scatter(fpr_alg[sample_thresh],tpr_alg[sample_thresh], c = 'y', marker = "P", s=100, label='kNN')
        plt.plot(fpr_alg[sample_thresh],tpr_alg[sample_thresh], c = 'y', marker = "P", markersize=10, label='kNN')
        auc_all[j,5] = metrics.roc_auc_score(y, alg_scores)
        
        print('\n******PCA*******\n')
        clf = PCA()
        clf.fit(X)
        alg_scores = clf.decision_scores_ 
        fpr_alg, tpr_alg, thresholds_alg = metrics.roc_curve(y, alg_scores, pos_label=1)
        #plt.plot(fpr_alg,tpr_alg, 'k')
        thresh_len = len(fpr_alg)
        sample_thresh = np.int_( [k * thresh_len/10 for k in range(10)] )
        sample_thresh = np.concatenate( [sample_thresh, np.asarray([thresh_len-1]) ])
        #plt.scatter(fpr_alg[sample_thresh],tpr_alg[sample_thresh], c = 'k', marker = ">", s=100, label='PCA')
        plt.plot(fpr_alg[sample_thresh],tpr_alg[sample_thresh], c = 'b', marker = ".", markersize=10, label='PCA')
        auc_all[j,6] = metrics.roc_auc_score(y, alg_scores)
        
    file_name = 'experiment_results/' + datasets[i] + '.pdf'
    
    for k in range(0,7):
        print('{:.4f}\t'.format( auc_all[j,k] ))
    print('\n')

    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend(loc='best')
    plt.title('ROC curve')
    # plt.show()
    f.savefig(file_name, bbox_inches='tight')
                
    file_name = 'experiment_results/' + datasets[i] + '_results_plot.mat'
    sio.savemat(file_name, {'auc_all':auc_all})

    