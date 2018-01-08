from sklearn.model_selection import cross_val_score, KFold
from hyperopt import fmin, tpe, Trials, space_eval, rand, anneal, mix, partial


class Search(object):
    def __init__(self, model, data, paramSpace, outputFolder, options=None):

        # attributes set by user
        self.model = model
        self.data = data
        self.paramSpace = paramSpace
        self.outputFolder = outputFolder
        self.options = options

        # attributes set by class methods
        self.trials = None
        self.best_model = None
        self.best_params = None
        self.best = None

    # the objective to minimize
    def objective(self, params):

        self.model.set_params(**params)
        X = self.data[0]
        y = self.data[1]

        # options
        n_splits = 5
        shuffle = True
        scoring = 'roc_auc'

        fmin = self.__fmin__(self.model, X, y, n_splits, shuffle, scoring)

        return fmin

    def __fmin__(self, model, X, y, n_splits,
                 shuffle, scoring, verbose=1, n_jobs=1):
        shuffle = KFold(n_splits=n_splits, shuffle=shuffle)
        score = cross_val_score(estimator=model, X=X, y=y,
                                cv=shuffle, scoring=scoring,
                                verbose=verbose, n_jobs=n_jobs)
        return 1.0 - score.mean()

    def run(self):
        self.trials = Trials()

        # could be shifted to options
        mix_algo = partial(mix.suggest, p_suggest=[
            (0.1, rand.suggest),
            (0.2, anneal.suggest),
            (0.7, tpe.suggest)
        ])

        self.best = fmin(self.objective,
                         self.paramSpace,
                         algo=mix_algo,
                         max_evals=5,
                         trials=self.trials,
                         verbose=True)
        self.best_params = space_eval(self.paramSpace, self.best)
        print(self.trials.best_trial)
        return self.best_params
