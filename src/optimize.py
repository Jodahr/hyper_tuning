# import params as pm
from sklearn.model_selection import cross_val_score, KFold
from hyperopt import fmin, tpe, Trials, space_eval
import pickle


class Search:
    def __init__(self, model, space, X, y, score='roc_auc'):
        self.model = model
        self.space = space
        self.X = X
        self.y = y
        self.score = score
        self.trial = None
        self.best_model = None
        
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
                    max_evals=50,
                    trials=trials,
                    verbose=True)
        best_params = space_eval(self.space, best)
        print(trials.best_trial)
        print(best_params)
        return best_params

    def run_trials(self):
        trials_step = 1
        max_trials = 5

        try:
            trials = pickle.load(open("../output/my_model.hyperopt"), "rb")
            print("found and load\n")
            max_trials = len(trials.trials) + trials_step
            print("Rerunning from {} trials to {} (+{}) trials"
                  .format(len(trials.trials), max_trials, trials_step))
        except:
            trials = Trials()
        best = fmin(self.objective,
                    self.space,
                    algo=tpe.suggest,
                    max_evals=max_trials,
                    trials=trials,
                    verbose=True)
        print("Best:", best)

        #   with open(_model + ".hyperopt", "wb") as f:
        # pickle.dump(trials, f)
        with open("../output/my_model.hyperopt", "wb") as f:
            pickle.dump(trials, f)

    def inf_search(self):
        try:
            while True:
                self.run_trials()
        except KeyboardInterrupt:
            pass
            
