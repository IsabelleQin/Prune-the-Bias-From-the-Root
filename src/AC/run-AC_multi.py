#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
sys.path.append('./')
from utils.verif_utils import *
import os
import numpy as np
import json
from fairlearn.metrics import *

df, X_train, y_train, X_test, y_test = load_adult_ac1()

model_dir = './models/adult/'
model_files = os.listdir(model_dir)
feature_names = list(df.columns)
sens_att = ['sex', 'race']
sens_idx = [feature_names.index(s) for s in sens_att]
subgroups = [[i, j] for i in range(2) for j in range(5)]

result_dict = {
    model_file.split('.')[0]:{
        'Original': {'Acc': [], 'TPR': {}},
        'Corrected': {'Acc': [], 'TPR': {}},
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
    outputs = model(model.inputs)

    predict_fn = lambda x: np.array([model.predict(x.reshape(1, -1, 13)).astype(float).flatten(), 1-model.predict(x.reshape(1, -1, 13)).astype(float).flatten()]).T

    y_pred = np.array([1 if item>=0.5 else 0 for item in model.predict(X_test.reshape(-1, 1, 13)).flatten()])
    sens_features = X_test[:, sens_idx]
    
    # Calculate the subgroup true positive before correction
    for i, subgroup in enumerate(subgroups):
        subgroup_idx = np.where((sens_features[:, 0] == subgroup[0]) & (sens_features[:, 1] == subgroup[1]))[0]
        result_dict[model_name]['Original']['TPR'][str(subgroup)] = round(float(sum(y_test[subgroup_idx] == y_pred[subgroup_idx])/len(subgroup_idx)), 3)
    
    # Results before change
    result_dict[model_name]['Original']['Acc'] = round(accuracy_score(y_test, y_pred), 3)

    names = [layer.name for layer in model.layers]
    first_layer = model.get_layer(names[0]).get_weights()

    # Modify the sensitive attribute change
    first_layer[0][sens_idx] = np.zeros(first_layer[0].shape[1])
    model.get_layer(names[0]).set_weights(first_layer)
    

    y_pred = np.array([1 if item>=0.5 else 0 for item in model.predict(X_test.reshape(-1, 1, 13)).flatten()])
    result_dict[model_name]['Corrected']['Acc'] = round(accuracy_score(y_test, y_pred), 3)
    
    # Calculate the subgroup true positive after correction
    for i, subgroup in enumerate(subgroups):
        subgroup_idx = np.where((sens_features[:, 0] == subgroup[0]) & (sens_features[:, 1] == subgroup[1]))[0]
        result_dict[model_name]['Corrected']['TPR'][str(subgroup)] = round(float(sum(y_test[subgroup_idx] == y_pred[subgroup_idx])/len(subgroup_idx)), 3)

with open('results/AC/prune_multiple_attributes.json', "w") as file:
    json.dump(result_dict, file, indent=4)
    