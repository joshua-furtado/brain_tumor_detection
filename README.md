# Brain Tumor Detection

## Description

**brain_tumor_detection** uses a Convolutional Neural Network (CNN) model to classify MRI brain scans. It's performance is compared against a Support Vector Machine classifier as benchmark.

## Dependencies

stock_price_indicator:

- NumPy
- OpenCV
- Matplotlib
- sklearn
- Tensorflow 2
- keras-tuner

## Folder Descriptions

1. [data/dataset](https://github.com/joshua-furtado/brain_tumor_detection/tree/main/data/dataset)
	- Contains 2 folders "yes" and "no" of brain scan images with and without tumor

2. [notebooks](https://github.com/joshua-furtado/brain_tumor_detection/tree/main/notebooks)
	- Contains Jupyter notebooks (names prefixed in sequential order)

## Notebook Descriptions

1. [1_neural_network_model.ipynb](https://github.com/joshua-furtado/brain_tumor_detection/blob/main/notebooks/1_neural_network_model.ipynb)
	- Build, train and test a CNN to classify brain scan images with and without tumor

2. [2_neural_network_model_with_data_augmentation.ipynb](https://github.com/joshua-furtado/brain_tumor_detection/blob/main/notebooks/2_neural_network_model_with_data_augmentation.ipynb)
	- Build, train and test a CNN using an augmented dataset to increase sample size

3. [3_neural_network_hyperparameter_tuning.ipynb](https://github.com/joshua-furtado/brain_tumor_detection/blob/main/notebooks/3_neural_network_hyperparameter_tuning.ipynb)
	- Improve upon the base CNN model by evaluating a range of values in the hyperparameter space using cross validation 

4. [4_SVC.ipynb](https://github.com/joshua-furtado/brain_tumor_detection/blob/main/notebooks/4_SVC.ipynb)
	- Build and evaluate a Support Vector Classifier as a benchmark to compare our model performance

## Results

The results of this work can be found [here](https://joshua-furtado.medium.com/detecting-brain-tumors-using-deep-learning-c2cd46d3cc1a).

## Authors

**Joshua Furtado**
