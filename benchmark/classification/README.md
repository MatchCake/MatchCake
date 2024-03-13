# Benchmarking of the Non-Interacting Fermionic Device

In this folder, you can find all the scripts used to generate the results from the paper 
[Fermionic Machine learning](). The scripts are organized in the following way:


### [classification_pipeline.py](classification_pipeline.py)

In this file you can find the pipeline used to load the dataset, preprocess it, build the quantum kernels,
train the SVM with the quantum kernels and evaluate the performance of the models.


### [main.py](main.py)

This file is used to run the pipeline for a given dataset and a given list of kernels.



### [dataset_complexity.py](dataset_complexity.py)

This file is used to generate the results of the dataset complexity analysis. It will create
a ensemble of classification pipelines with different number of qubits for a given dataset and a given list of kernels.
It will also create the associated plots.


### [kernels.py](kernels.py)

This file contains the definition of the quantum kernels used in the paper.


### [utils.py](utils.py)

This file contains some utility functions used in the other scripts.


### [requirements.txt](requirements.txt)

This file contains the list of the required packages to run the scripts.
