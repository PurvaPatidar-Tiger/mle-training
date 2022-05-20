# Median housing value prediction

The housing data can be downloaded from https://raw.githubusercontent.com/ageron/handson-ml/master/. The script has codes to download the data. We have modelled the median house value on given housing data.

The following techniques have been used:

 - Linear regression
 - Decision Tree
 - Random Forest

## Installation:
### Prerequisites:
`$ conda activate mle-dev`

### Setup:
For editable install:
`$ pip install -e .`

## Run code:
### To download and process data, To train the models and To score trained models:
`run the main.py file`

### Note:
You can get information on command line arguments for each of the above scripts using `-h` or `--help`. For example:

## Steps performed:
 - We prepared and cleaned the data.
 - We checked and imputed missing values.
 - Features are generated and the variables are checked for correlation.
 - Multiple sampling techinuqies are evaluated. The data set is split into train and test.
 - All the above said modelling techniques are tried and evaluated. The final metric used to evaluate is mean squared error.


