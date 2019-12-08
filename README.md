# PIDforest
Code for the PIDForest algorithm for anomaly detection.

The PIDForest algorithm is based on the Partial Identification framework for anomaly detection. Partial Identification captures the intuition that anomalies are easy to distinguish from the overwhelming majority of points by relatively few attribute values. PIDScore is a geometric anomaly score based on this framework, and it measures the minimum density of data points over all subcubes containing the point. PIDForest is a random forest based algorithm that finds anomalies based on PIDScore. 

The [accompanying paper](https://papers.nips.cc/paper/9710-pidforest-anomaly-detection-via-partial-identification.pdf) shows that PIDForest performs favorably in comparison to several popular anomaly detection methods, across a broad range of benchmarks. PIDForest also provides a succinct explanation for why a point is labelled anomalous, by providing a set of features and ranges for them which are relatively uncommon in the dataset.
