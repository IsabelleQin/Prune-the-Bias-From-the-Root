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

df, X_train, y_train, X_val, y_val, X_test, y_test = load_compas()
model_dir = './models/compas/'
model_files = [f for f in os.listdir(model_dir) if '.h5' in f]

X_train_df = pd.DataFrame(np.concatenate([X_train, y_train.reshape(-1, 1)], axis=1), columns=df.columns)
X_test_df = pd.DataFrame(np.concatenate([X_test, y_test.reshape(-1, 1)], axis=1), columns=df.columns)

dataset_test = StandardDataset(X_test_df, label_name='Two_yr_Recidivism',
                 favorable_classes=[0, 0],
                 protected_attribute_names=['Age', 'Race'],
                 privileged_classes=[[1], [0]],
                 instance_weights_name=None,
                 categorical_features=['Number_of_Priors', 'score_factor', 'Female', 'Misdemeanor'],
                 na_values=[], custom_preprocessing=None)

dataset_train = StandardDataset(X_train_df, label_name='Two_yr_Recidivism',
                 favorable_classes=[0, 0],
                 protected_attribute_names=['Age', 'Race'],
                 privileged_classes=[[1], [0]],
                 instance_weights_name=None,
                 categorical_features=['Number_of_Priors', 'score_factor', 'Female', 'Misdemeanor'],
                 na_values=[], custom_preprocessing=None)

result_dict = {
    model_file.split('.')[0]:{
        'Original': {'Acc': [], 'DP': [], 'EO': []},
        'Corrected': {'Acc': [], 'DP': [], 'EO': []},
        } for model_file in model_files
}

feature_names = list(df.columns)
sens_att = 'Race'
sens_idx = feature_names.index(sens_att)
for model_file in model_files:
    if not model_file.endswith('.h5'):
        continue;
    print('==================  STARTING MODEL ' + model_file)
    model_name = model_file.split('.')[0]
    if model_name == '':
        continue
    model = load_model(model_dir + model_file)
    
    dataset_train_pred = dataset_train.copy()
    res = model.predict(X_train.reshape(1, -1, 6)).flatten()
    y_pred = [[1] if s>=0.5 else [0] for s in res]
    dataset_train_pred.scores = res.reshape(-1,1)
    dataset_train_pred.labels = np.array(y_pred)

    dataset_test_pred = dataset_test.copy()
    names = [layer.name for layer in model.layers]
    first_layer = model.get_layer(names[0]).get_weights()

    # Modify the sensitive attribute change
    first_layer[0][sens_idx] = np.zeros(first_layer[0].shape[1])
    model.get_layer(names[0]).set_weights(first_layer)
    res = model.predict(X_test.reshape(1, -1, 6)).flatten()
    y_pred = [[1] if s>=0.5 else [0] for s in res]
    dataset_test_pred.labels = np.array(y_pred)
    dataset_test_pred.scores = res.reshape(-1,1)
    privileged_groups = [{'Race': 0}]
    unprivileged_groups = [{'Race': 1}]
    
    # Upper and lower bound on the fairness metric used
    ROC = RejectOptionClassification(unprivileged_groups=unprivileged_groups, 
                                 privileged_groups=privileged_groups, metric_name='Equal opportunity difference',)
    
    
    ROC = ROC.fit(dataset_train, dataset_train_pred)
    dataset_test_pred_post = ROC.predict(dataset_test_pred)
    
    sens_features = X_test[:, sens_idx]
    
    # Results before change
    result_dict[model_name]['Original']['Acc'] = round(accuracy_score(y_test, dataset_test_pred.labels), 3)
    result_dict[model_name]['Original']['DP'] = round(demographic_parity_difference(y_test, dataset_test_pred.labels, sensitive_features=sens_features), 3)
    result_dict[model_name]['Original']['EO'] = round(equal_opportunity_difference(y_test, dataset_test_pred.labels, sensitive_features=sens_features), 3)
    # Results after change
    result_dict[model_name]['Corrected']['Acc'] = round(accuracy_score(y_test, dataset_test_pred_post.labels), 3)
    result_dict[model_name]['Corrected']['DP'] = round(demographic_parity_difference(y_test, dataset_test_pred_post.labels, sensitive_features=sens_features), 3)
    result_dict[model_name]['Corrected']['EO'] = round(equal_opportunity_difference(y_test, dataset_test_pred_post.labels, sensitive_features=sens_features), 3)

with open('results/compas/ROC.json', "w") as file:
    json.dump(result_dict, file, indent=4)