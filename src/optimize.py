# import params as pm
from sklearn.model_selection import cross_val_score, KFold
from hyperopt import fmin, tpe, Trials, space_eval


class Search:
    def __init__(self, model, space, X, y, score='roc_auc'):
        self.model = model
        self.space = space
        self.X = X
        self.y = y
        self.score = score
        
    def objective(self, params):
        print("\n1\n")
        self.model.set_params(**params)
        print("2\n")
        shuffle = KFold(n_splits=3, shuffle=True)
        print("3\n")
        score = cross_val_score(self.model, self.X, self.y,
                                cv=shuffle, scoring='roc_auc',
                                n_jobs=1, verbose=1)
        print("4\n")
        print(score.mean())
        return 1-score.mean()

    def run(self):
        trials = Trials()
        best = fmin(self.objective,
                    self.space,
                    algo=tpe.suggest,
                    max_evals=250,
                    trials=trials,
                    verbose=True)
        best_params = space_eval(self.space, best)
        print(best_params)
        return best_params
