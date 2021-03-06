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
Afterwards ,you can config your model parameters and the search space in a json file.
The remaining configs can be set in an init file.

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
* filename for hyperopt trial object needs to be changed when input is changed; so, use argparser for that purpose
* change your print commands to print to stderr or logger
* add a HACKING file for your ideas
* add a option to train the model after all iterations are done; not just when you quit the session with CTRL+C
* nested cross validation
* generalize for multiclass and regression
* generalize for unsupervised learning
* save retrained models
* add summary metric results
* add templates for stacking models, etching
* add option to create empty parameterdict
* reconsider init and paramsinput
* test for house pricing and a classification model
* add hyperopt unlearn options
