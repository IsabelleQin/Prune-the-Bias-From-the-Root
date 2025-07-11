from scipy.stats import *
import json

datasets = ['AC', 'BM', 'GC', 'compas']
approaches = ['CalibratedEqOdds', 'EqOdds', 'ROC', 'prune_attributes']

ttest_results = {dataset:{approach:{} for approach in approaches} for dataset in datasets}


for dataset in datasets:
    for approach in approaches:
        # List for the data
        original_acc, original_dp, original_eo = [], [], []
        corrected_acc, corrected_dp, corrected_eo = [], [], []

        # Load the results for the current dataset and approach
        with open(f'results/%s/%s.json'%(dataset, approach), 'r') as file:
            results = json.load(file)
            for model in results.keys():
                original_acc.append(results[model]['Original']['Acc'])
                original_dp.append(results[model]['Original']['DP'])
                original_eo.append(results[model]['Original']['EO'])
                corrected_acc.append(results[model]['Corrected']['Acc'])
                corrected_dp.append(results[model]['Corrected']['DP'])
                corrected_eo.append(results[model]['Corrected']['EO'])
            
        # Perform the t-test for each metric
        acc_ttest = ttest_ind(original_acc, corrected_acc)
        dp_ttest = ttest_ind(original_dp, corrected_dp)
        eo_ttest = ttest_ind(original_eo, corrected_eo)

        ttest_results[dataset][approach] = {
            'Acc': round(acc_ttest.pvalue, 3),
            'DP': round(dp_ttest.pvalue, 3),
            'EO': round(eo_ttest.pvalue, 3)
        }

with open('results/single_att_ttest.json', "w") as file:
    json.dump(ttest_results, file, indent=4)