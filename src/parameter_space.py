from hyperopt import hp
import numpy as np

# for more complex search spaces
# here you can also define different models
# (similar thing is done in hyperopt-sklearn)
# for more information see link in README file
def complexSpace(searchDict):
    # define your own complex space
    try:
        # combined
        searchDict['bbc__base_estimator__max_depth'] = hp.choice('bbc__base_estimator__max_depth', [
            hp.choice('c1',
                      [None]),
            hp.choice('c2',
                      [x for x in range(1, 50+1, 2)])
        ])
        searchDict['bbc__base_estimator__max_features'] = hp.choice('bbc__base_estimator__max_features',
                                                   ['sqrt', 'log2',
                                                    None])
        searchDict['bbc__ratio'] = hp.choice('bbc__ratio',
                                             ['auto', 'not minority',
                                              'majority', 'all'])
        searchDict['bbc__replacement'] = hp.choice('bbc__replacement',
                                                   [True, False])
        searchDict['bbc__max_features'] = hp.choice('bbc__max_features',
                                                    [
                                                       hp.choice('c3',
                                                                 [1.0]),
                                                       hp.uniform('c4',
                                                                  0.1, 0.9)
                                                    ])
    except:
        print("not working")
    return searchDict
