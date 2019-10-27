Code for PIDForest can be found in the scripts/ folder
The following scripts return the AUC for the experiments:

run_classication_pyod: Runs classfication based datasets on PyOD algorithms (kNN, PCA)
run_timeseries_pyod: Runs timeseries based datasets on PyOD algorithms (kNN, PCA)
run_classication_sklearn_pidforest_alg: Runs classfication based datasets on sklearn algorithms (LOF, SVM, IF) and PIDForest
run_timeseries_pyod: Runs timeseries based datasets on sklearn algorithms (LOF, SVM, IF) and PIDForest
run_classication_rrcf: Runs classfication based datasets on rrcf
run_timeseries_pyod: Runs timeseries based datasets on rrcf

The get_roc_series and get_roc_classification scripts generate ROC curves for the datasets for all algorithms.

The ../synthetic_experiments folder has jupyter notebooks for running the synthetic experiments (mixture of Gaussians, timeseries and masking).

To run the above scripts, the following packages are needed:

scikit-learn
pandas
numpy
rrcf (https://github.com/kLabUM/rrcf)
PyOD (https://github.com/yzhao062/pyod)