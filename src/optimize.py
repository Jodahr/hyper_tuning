# import params as pm
from sklearn.model_selection import cross_val_score, KFold
from hyperopt import fmin, tpe, Trials, space_eval, rand, anneal, mix, partial
import pickle
from sklearn.metrics import roc_auc_score
import numpy as np
from geographiclib.geodesic import Geodesic
import math
geod = Geodesic.WGS84
import logging


class Search:
    def __init__(self, model, space, X, y, X_test, y_test, score='roc_auc'):
        self.model = model
        self.space = space
        self.X = X
        self.y = y
        self.X_test = X_test
        self.y_test = y_test
        self.score = score
        self.trials = None
        self.best_model = None
        self.best_params = None
        self.best = None
        
    def objective(self, params):
        self.model.set_params(**params)
        shuffle = KFold(n_splits=3, shuffle=True)
        score = cross_val_score(self.model, self.X, self.y,
                                cv=shuffle, scoring='roc_auc',
                                n_jobs=1, verbose=1)
        # print(score.mean())
        return 1-score.mean()

    def run(self):
        self.trials = Trials()

        mix_algo = partial(mix.suggest, p_suggest=[
            (0.1, rand.suggest),
            (0.2, anneal.suggest),
            (0.7, tpe.suggest)
        ])

        self.best = fmin(self.objective,
                         self.space,
                         algo=mix_algo,
                         max_evals=50,
                         trials=self.trials,
                         verbose=True)
        self.best_params = space_eval(self.space, self.best)
        print(self.trials.best_trial)
        # print(self.best_params)
        return self.best_params

    def run_trials(self):
        trials_step = 1
        max_trials = 3
        try:
            self.trials = pickle.load(open("../output/my_model_gb_bbc3.hyperopt",
                                      "rb"))
            print("found and load\n")
            max_trials = len(self.trials.trials) + trials_step
            print("Rerunning from {} trials to {} (+{}) trials"
                  .format(len(self.trials.trials), max_trials, trials_step))
        except:
            self.trials = Trials()
        self.best = fmin(self.objective,
                         self.space,
                         algo=tpe.suggest,
                         max_evals=max_trials,
                         trials=self.trials,
                         verbose=True)
        #print("Best:", self.best)
        self.best_params = space_eval(self.space, self.best)
        logging.debug('best params\n: {} \n'.format(self.best_params))
        #   with open(_model + ".hyperopt", "wb") as f:
        # pickle.dump(trials, f)
        with open("../output/my_model_gb_bbc3.hyperopt", "wb") as f:
            pickle.dump(self.trials, f)

    def inf_search(self, n=100):
        try:
            i = 0
            while i < n:
                print("run {}/{}".format(i, n))
                self.run_trials()
                i += 1
        except KeyboardInterrupt:
            #print(self.trials.best_trial)
            #print(self.trials.results)
            #print(self.trials.trials[:2])
            print(self.best_params)
            #print(self.best)
            self.retrain_best_model()
            self.final_score()
            pass

    def retrain_best_model(self):
        print("retraining model with full data ...\n")
        self.model.set_params(**self.best_params)
        self.model.fit(self.X, self.y)
        print("...done\n")
        return self.model

    # only binary classification or maybe also multiclass
    def final_score(self):
        y_score = self.model.predict_proba(self.X_test)
        roc = roc_auc_score(self.y_test, y_score[:, 1])
        print(roc)
    
    # def sc(self):
    #     n_classes = len(self.y[0])
    #     print(n_classes)
    #     fpr = dict()
    #     tpr = dict()
    #     roc_auc = dict()
    #     y_score = self.model.predict_proba(self.X_test)
    #     for i in range(n_classes):
    #         print(self.y_test[:,1])
    #         print(y_score)
    #         fpr[i], tpr[i], _ = roc_curve(self.y_test[:, i], y_score[:, i])
    #         print("hello")
    #         roc_auc[i] = auc(fpr[i], tpr[i])

    #         # Compute micro-average ROC curve and ROC area
    #     fpr["micro"], tpr["micro"], _ = roc_curve(self.y_test.ravel(),
    #                                               y_score.ravel())
    #     roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    #     print(roc_auc["micro"])
