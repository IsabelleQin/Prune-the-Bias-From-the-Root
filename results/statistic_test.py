from scipy.stats import *
import toleranceinterval as ti
import json

datasets = ['AC', 'BM', 'GC', 'compas']
approaches = ['CalibratedEqOdds', 'EqOdds', 'ROC', 'prune_attributes']

ttest_results = {dataset:{approach:{} for approach in approaches} for dataset in datasets}
p=0.95
gamma = 0.95

for dataset in datasets:
    for approach in approaches:
        # List for the data
        original_acc, original_dp, original_eo = [], [], []
        corrected_acc, corrected_dp, corrected_eo = [], [], []
        acc_gap = []

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

                acc_gap = [o-c for o, c in zip(original_acc, corrected_acc)]
            
        # Perform the t-test for each metric
        acc_ttest = ttest_ind(original_acc, corrected_acc)
        dp_ttest = ttest_ind(original_dp, corrected_dp)
        eo_ttest = ttest_ind(original_eo, corrected_eo)
        bounds = ti.oneside.normal(acc_gap, g=gamma, p=p)

        ttest_results[dataset][approach] = {
            'Acc': round(acc_ttest.pvalue, 3),
            'Acc upper': round(bounds[0], 3),
            'DP': round(dp_ttest.pvalue, 3),
            'EO': round(eo_ttest.pvalue, 3)
        }

with open('results/single_att_ttest.json', "w") as file:
    json.dump(ttest_results, file, indent=4)
