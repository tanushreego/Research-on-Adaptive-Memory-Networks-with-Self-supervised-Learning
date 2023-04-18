# AMSL

# Project Abstract :

Unsupervised anomaly detection tries to construct models to effectively detect undetected anomalies by just training on normal data. Although prior reconstruction-based approaches have made significant strides, their capacity to generalise is constrained by two major obstacles. First off, the training dataset is restricted to only normal patterns, which hinders the generalizability of the model. Second, it is frequently difficult to maintain the diversity of normal patterns because the feature representations learnt by previous models frequently lack representativeness. To overcome these difficulties and improve the generalizability of unsupervised anomaly detection, we suggest a method termed Adaptive Memory Network with Self-supervised Learning. The AMSL comprises a self-supervised learning module to learn generic normal patterns based on the convolutional auto encoder structure and an adaptive memory fusion module to learn rich feature representations.

# Installation
### Requirements
* Python == 3.6
* Cuda == 9.1
* Keras ==2.2.2
* Tensorflow ==1.8.0


# Usage
We use DASADS dataset(refer to [UCI](https://archive.ics.uci.edu/ml/datasets/daily+and+sports+activities)) as demo example. 

## Data Preparation
```
# run preprocessing.py to generate normal and abnormal datasets.

data
 |-DASADS
 | |-a01  
 | | |-p1  
 | | | |-s01.txt
 | |-...
 | |-a09
 |-generate_dataset
 | |-normal.npy
 | |-abnormal.npy

 # run transformation.py to transform all the data and then separate them into training and testing datasets.

 data
 |-generate_dataset
 | |-normal.npy
 | |-abnormal.npy
 |-transform_dataset
 | |-train_dataset
 | | |-data_raw_train.npy  # raw data
 | | |-data_no_train.npy   # noise data
 | | |-data_ne_train.npy   # negated data
 | | |-data_op_train.npy   # opposite_time data
 | | |-data_pe_train.npy   # permuted data
 | | |-data_sc_train.npy   # scale data
 | | |-data_ti_train.npy   # time_warp data
 | |-test_dataset
 | | |-normal data
 | | | |-data_raw_test.npy  # raw data
 | | | |-data_no_test.npy   # noise data
 | | | |-data_ne_test.npy   # negated data
 | | | |-data_op_test.npy   # opposite_time data
 | | | |-data_pe_test.npy   # permuted data
 | | | |-data_sc_test.npy   # scale data
 | | | |-data_ti_test.npy   # time_warp data
 | | |-abnormal data
 | | | |-data_raw_abnormal.npy  # raw data
 | | | |-data_no_abnormal.npy   # noise data
 | | | |-data_ne_abnormal.npy   # negated data
 | | | |-data_op_abnormal.npy   # opposite_time data
 | | | |-data_pe_abnormal.npy   # permuted data
 | | | |-data_sc_abnormal.npy   # scale data
 | | | |-data_ti_abnormal.npy   # time_warp data

```

## Run

### Train model

You can get results of the MSE loss after running train.py. 

```

results
 |-train_normal_loss_sum_mse.csv  #the MSE loss of training data
 |-normal_loss_sum_mse.csv  #the MSE loss of normal data in the testing dataset
 |-abnormal_loss_sum_mse.csv #the MSE loss of abnormal data in the testing dataset

```

### Evaluation

Run evaluate.py to compute the threshold by the MSE loss of training data and achieve the accuracy, precision, recall and F1 score of testing data.

