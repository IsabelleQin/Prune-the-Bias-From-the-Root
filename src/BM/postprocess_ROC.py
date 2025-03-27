#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
sys.path.append('./')
from utils.verif_utils import *
import os
import numpy as np
from fairlearn.metrics import *
from aif360.datasets import StandardDataset
from aif360.algorithms.postprocessing.reject_option_classification import RejectOptionClassification
import json

df, X_train, y_train, X_test, y_test = load_bank()

model_dir = './models/bank/'
model_files = os.listdir(model_dir)

X_train_df = pd.DataFrame(np.concatenate([X_train, y_train.reshape(-1, 1)], axis=1), columns=df.columns)
X_test_df = pd.DataFrame(np.concatenate([X_test, y_test.reshape(-1, 1)], axis=1), columns=df.columns)

dataset_test = StandardDataset(X_test_df, label_name='y',
                 favorable_classes=[1],
                 protected_attribute_names=['age'],
                 privileged_classes=[[1]],
                 instance_weights_name=None,
                 categorical_features=['job', 'marital', 'education', 'default', 
                                       'housing', 'loan', 'contact', 'month', 
                                       'day_of_week', 'poutcome'],
                 na_values=['?'], custom_preprocessing=None)

dataset_train = StandardDataset(X_train_df, label_name='y',
                 favorable_classes=[1],
                 protected_attribute_names=['age'],
                 privileged_classes=[[1]],
                 instance_weights_name=None,
                 categorical_features=['job', 'marital', 'education', 'default', 
                                       'housing', 'loan', 'contact', 'month', 
                                       'day_of_week', 'poutcome'],
                 na_values=['?'], custom_preprocessing=None)

sens_att = 'age'
feature_names = list(df.columns)
sens_idx = feature_names.index(sens_att)

result_dict = {
    model_file.split('.')[0]:{
        'Original': {'Acc': [], 'DP': [], 'EO': []},
        'Corrected': {'Acc': [], 'DP': [], 'EO': []},
        } for model_file in model_files
}

for model_file in model_files:
    if not model_file.endswith('.h5'):
        continue;
    print('==================  STARTING MODEL ' + model_file)
    model_name = model_file.split('.')[0]
    if model_name == '':
        continue
    model = load_model(model_dir + model_file, compile=False)
    # Manually define an input tensor to fix the missing input issue
    dataset_train_pred = dataset_train.copy()
    res = model.predict(X_train.reshape(1, -1, 16)).flatten()
    y_pred = [[1] if s>=0.5 else [0] for s in res]
    dataset_train_pred.scores = res.reshape(-1,1)
    dataset_train_pred.labels = np.array(y_pred)

    dataset_test_pred = dataset_test.copy()
    res = model.predict(X_test.reshape(1, -1, 16)).flatten()
    y_pred = [[1] if s>=0.5 else [0] for s in res]
    dataset_test_pred.labels = np.array(y_pred)
    dataset_test_pred.scores = res.reshape(-1,1)
    privileged_groups = [{'age': 1}]
    unprivileged_groups = [{'age': 0}]

    cpp = RejectOptionClassification(unprivileged_groups=unprivileged_groups, 
                                 privileged_groups=privileged_groups, )
    
    cpp = cpp.fit(dataset_train, dataset_train_pred)
    dataset_test_pred_post = cpp.predict(dataset_test_pred)
    
    sens_features = np.array([[1]if i == 1 else [0] for i in X_test[:, sens_idx]])
    
    # Results before change
    result_dict[model_name]['Original']['Acc'] = round(accuracy_score(y_test, dataset_test_pred.labels), 3)
    result_dict[model_name]['Original']['DP'] = round(demographic_parity_difference(y_test, dataset_test_pred.labels, sensitive_features=sens_features), 3)
    result_dict[model_name]['Original']['EO'] = round(equal_opportunity_difference(y_test, dataset_test_pred.labels, sensitive_features=sens_features), 3)

    
    result_dict[model_name]['Corrected']['Acc'] = round(accuracy_score(y_test, dataset_test_pred_post.labels), 3)
    result_dict[model_name]['Corrected']['DP'] = round(demographic_parity_difference(y_test, dataset_test_pred_post.labels, sensitive_features=sens_features), 3)
    result_dict[model_name]['Corrected']['EO'] = round(equal_opportunity_difference(y_test, dataset_test_pred_post.labels, sensitive_features=sens_features), 3)

with open('results/BM/ROC.json', "w") as file:
    json.dump(result_dict, file, indent=4)