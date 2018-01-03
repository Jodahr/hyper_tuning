from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import cross_val_score, KFold


class Search(object):
    def __init__(self, model, data, paramSpace, scoring='mse', **kwargs):
        self.model = model
        self.data = data
        self.paramSpace = paramSpace
        #self.__dict__.update(kwargs)

        
class Fmin(object):
    def __init__(self, model, X, y, scoring,
                 n_splits=3, shuffle=True, n_jobs=1, verbose=1):
                    self.model = model
                    self.X = X
                    self.y = y
                    self.scoring = scoring
                    self.n_splits = n_splits
                    self.shuffle = shuffle
                    self.verbose = verbose
                    self.n_jobs = n_jobs
                    
    def __fmin__(self):
        shuffle = KFold(n_splits=self.n_splits, shuffle=True, **self.kwargs)
        score = cross_val_score(estimator=self.model, X=self.X, y=self.y,
                                cv=shuffle, scoring=self.scoring,
                                verbose=self.verbose, n_jobs=self.n_jobs)
        return 1.0 - score.mean()
        
        
        
