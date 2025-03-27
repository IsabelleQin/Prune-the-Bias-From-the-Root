# Prune-the-Bias-From-the-Root
The experiment data, model, and results for "Prune the Bias From the Root: Bias Removal and Fairness Estimation by Muting Sensitive Attributes in Pre-trained DNN Models". 

## Dataset
To comprehensively understand the impact of sensitive attribute pruning, we select three commonly used fairness datasets, namely Bank Marketing (BM), Adult Census (AC), and German Credit (GC). 

## Models
The benchmark DNN model set for the three datasets is provided by a previous study by Biswas and Rajan, gathered from previous studies and Kaggle. To ensure correct implementation, we obtained the models and their parameters from the replication package provided by Biswas and Rajan. 

## Results 
The experiment results for single-attribute pruning and multi-attribute pruning, along with the results obtained with other post-processing bias removal methods, can be found under ```results``` folder. 

## Replicating the experiment
To replicate our experiment, please run the python files under ```src``` folder. 
