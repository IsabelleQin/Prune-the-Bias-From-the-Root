# Prune the Bias From the Root: Bias Removal and Fairness Estimation by Muting Sensitive Attributes in Pre-trained DNN Models
### The replication package for "Prune the Bias From the Root: Bias Removal and Fairness Estimation by Muting Sensitive Attributes in Pre-trained DNN Models". 

## Introduction
Attribute pruning is a simple yet effective post-processing technique that enforces individual fairness by zeroing out sensitive attribute weights in a pre-trained DNN’s input layer. To ensure the generalizability of our results, we conducted experiments on 32 models and 4 widely used datasets, and compared attribute pruning’s performance with 3 baseline post-processing methods (i.e., equalized odds, calibrated equalized odds, and ROC). In this study, we reveal the effectiveness of sensitive attribute pruning on small-scale DNN bias removal and discuss its usage in multi-attribute fairness estimation by answering the following research questions: 

### RQ1: How does single-attribute pruning perform in comparison to the existing post-processing methods?
By answering this research question, we aim to understand the accuracy and group fairness impact of single-attribute pruning on 32 models and compare them with 3 state-of-the-art post-processing methods. 

### RQ2: How does multi-attribute pruning impact and aid understanding of the original models?
By answering this research question, we investigate the accuracy impact of multi-attribute pruning on 24 models. Further, we investigate the prediction change brought by attribute pruning on different subgroups and discuss their implications on multi-attribute fairness estimation. 

## Dependencies
- Python >= 3.9
- numpy == 1.24.4
- fairlearn == 0.12.0
- aif360 == 0.6.1
- scikit-learn == 1.6.1
- tensorflow == 2.14.0
- pandas == 2.0.3
- scipy == 1.13.1

## Dataset
To comprehensively understand the impact of sensitive attribute pruning, we select four commonly used fairness datasets collected from different domains, namely Bank Marketing (BM), German Credit (GC), Adult Census (AC), and COMPAS. We select the four datasets because they provide a wide range of corresponding pre-trained models used in existing research. The introduction to the datasets is as follows:

**Bank Marketing (BM):** The [Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing) dataset consists of marketing data from a Portuguese bank, containing 45,222 instances with 16 attributes, and the biased attribute identified is **age**. The objective is to classify whether a client will subscribe to a term deposit.

**German Credit (GC):** The [German Credit](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data) dataset includes 1,000 instances of individuals who have taken credit from a bank, each described by 20 attributes, with two sensitive attributes, **sex and age**; the single sensitive attribute to be evaluated in RQ1 is **age**, given that the subgroup positive rate difference (i.e., historical bias in the label) on this sensitive attribute is higher than **sex**. The task is to classify the credit risk of an individual. 

**Adult Census (AC):** The [Adult Census](https://archive.ics.uci.edu/dataset/2/adult) dataset comprises United States census data from 41,188 individuals after empty entry removal, with 13 attributes. The sensitive attributes in the dataset are **sex and race**; the single sensitive attribute to be evaluated in RQ1 is **sex**. The goal is to predict whether an individual earns more than $50,000 per year.

**COMPAS:** The [COMPAS](https://mlr3fairness.mlr-org.com/reference/compas.html) (Correctional Offender Management Profiling for Alternative Sanctions) dataset is collected from a system widely used for criminal recidivism risk prediction. The sensitive attributes in the dataset are **race and age**; to keep aligned with previous research, the single sensitive attribute to be evaluated in RQ1 is **race**. The goal is to predict whether an individual will reoffend in the future.

## Experiments
To replicate the experiments, run the code in the ```src``` folder, the sub-folders contain the code for implementing the post-processing methods on each dataset. To obtain the basic results, run all the codes in each folder. The results will be stored in the ```results``` folder; we also provide the code for statistical analyses (i.e., paired t-test) under this folder. To conduct the statistical analyses, run ```statistic_test.py``` and check the results in ```single_att_ttest.json```.

### RQ1: How does single-attribute pruning perform in comparison to the existing post-processing methods?
While ensuring individual fairness on the single attribute, attribute pruning **will not significantly** impact accuracy. It preserved the highest post-processing accuracy among the four methods on 23 out of 32 models. It can also improve the two group accuracies in general, but its improvements are insignificant and not always optimal in comparison to the other three methods. Further, given the theoretical difference between individual fairness and group fairness, attribute pruning may even harm group fairness when the observed dataset is not comprehensive enough to cover the whole data space. 

![image](./tables/rq1_res.png?raw=true)

![image](./tables/rq1_ttest.png?raw=true)

### RQ2: How does multi-attribute pruning impact and aid understanding of the original models?
According to our experiment on 24 models, multi-attribute pruning can also retain a certain level of accuracy while enhancing individual fairness, as shown in the table below. 

![image](./tables/rq2_acc_res.png?raw=true)

It can also be used to estimate multi-attribute group fairness in models with similar original accuracy based on the TPR difference before and after pruning the sensitive attributes. To illustrate the assessment process, we select two models, AC3 and AC10, which share the same original accuracy of 0.845 before attribute pruning. Their TPRs on different [sex, race] subgroups are shown in the following table. This information is useful for practitioners in choosing models to meet specific requirements (e.g., general fairness or protection on specific subgroups), especially when the models share the same accuracy.

![image](./tables/rq2_TPR_diff.png?raw=true)

## Folder Structure
```
├── data # The 4 datasets used in the study
├── models # Model files for the 32 models included in our experiment
├── results # Results for RQ1 and RQ2
    ├── AC
    ├── BM
    ├── GC
    ├── compas
    ├── single_att_ttest.json # Statistical analysis results
    └── statistic_test.py
├── src # Codes for implementing the post-processing methods on each dataset
    ├── AC
    ├── BM
    ├── GC
    └── compas
├── utils
├── tables 
└── README.md
```
