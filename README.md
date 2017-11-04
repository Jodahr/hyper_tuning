# hyper_tuning

This program can be used to tune different machine learning models.

## dependencies:

* numpy
* sklearn
* hyperopt
* hyperopt-sklearn
* pandas
* ...

It will use hyperopt to optimize different scores like roc_auc, etc.
All you need is a model pipeline object saved in pkl format and the input data.
Afterwards you can config your model parameters and the search space in a json file.
The remaining congifs can be set in a init file.

## starting point:
* load data, model and creating init file
* tune roc_auc for classification model
* save trial and best model
* ...
* first we add options for hyperopt itself
* hyperopt-sklearn will be added later