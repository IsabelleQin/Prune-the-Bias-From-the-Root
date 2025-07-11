#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
sys.path.append('./')
from utils.verif_utils import *
import os
import numpy as np
from fairlearn.metrics import *
import json

df, X_train, y_train, X_test, y_test = load_german()
model_dir = './models/german/'
model_files = os.listdir(model_dir)

sens_att = 'sex'
feature_names = list(df.columns.drop('credit'))
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
    sens_features = X_test[:, sens_idx]

    # Results before change
    result_dict[model_name]['Original']['Acc'] = round(accuracy_score(y_test, y_pred), 3)
    result_dict[model_name]['Original']['DP'] = round(demographic_parity_difference(y_test, y_pred, sensitive_features=sens_features), 3)
    result_dict[model_name]['Original']['EO'] = round(equal_opportunity_difference(y_test, y_pred, sensitive_features=sens_features), 3)

    names = [layer.name for layer in model.layers]
    first_layer = model.get_layer(names[0]).get_weights()

    # Modify the sensitive attribute weight
    first_layer[0][sens_idx] = np.zeros(first_layer[0].shape[1])
    model.get_layer(names[0]).set_weights(first_layer)

    y_pred = [[1] if item>=0.5 else [0] for item in model.predict(X_test.reshape(-1, 1, 20)).flatten()]
    result_dict[model_name]['Corrected']['Acc'] = round(accuracy_score(y_test, y_pred), 3)
    result_dict[model_name]['Corrected']['DP'] = round(demographic_parity_difference(y_test, y_pred, sensitive_features=sens_features), 3)
    result_dict[model_name]['Corrected']['EO'] = round(equal_opportunity_difference(y_test, y_pred, sensitive_features=sens_features), 3)

with open('results/GC/prune_attributes.json', "w") as file:
    json.dump(result_dict, file, indent=4)