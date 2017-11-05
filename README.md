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


## Credit for Code snippets goes to:

* https://github.com/hyperopt/hyperopt/issues/267
* https://github.com/steventhornton/Hyperparameter-Tuning-with-hyperopt-in-Python
* https://github.com/hyperopt/hyperopt/wiki/FMin

## ideas
* try to add sth similar for spark models
* add multiclass option
* add unsupervised learning
* integrate hyperopt-sklearn
* use parser arguments in the code
* create a output folder containing trials and the best trained model
* generate plots and save them as serialized matplotlib objects
* use more OOP instead of so many functions