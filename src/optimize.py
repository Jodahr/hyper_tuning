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
        print(score.mean())
        return 1-score.mean()

    def run(self):
        self.trials = Trials()
        self.best = fmin(self.objective,
                         self.space,
                         algo=tpe.suggest,
                         max_evals=50,
                         trials=self.trials,
                         verbose=True)
        self.best_params = space_eval(self.space, self.best)
        print(self.trials.best_trial)
        print(self.best_params)
        return self.best_params

    def run_trials(self):
        trials_step = 1
        max_trials = 5
        try:
            print("1")
            self.trials = pickle.load(open("../output/my_model.hyperopt",
                                      "rb"))
            print("2")
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

        #   with open(_model + ".hyperopt", "wb") as f:
        # pickle.dump(trials, f)
        with open("../output/my_model.hyperopt", "wb") as f:
            pickle.dump(self.trials, f)

    def inf_search(self):
        try:
            while True:
                self.run_trials()
        except KeyboardInterrupt:
            #print(self.trials.best_trial)
            #print(self.trials.results)
            #print(self.trials.trials[:2])
            print(self.best_params)
            #print(self.best)
            pass
            
