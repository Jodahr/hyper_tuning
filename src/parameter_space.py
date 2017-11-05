from hyperopt import hp


# for more complex search spaces
# here you can also define different models
# (similar thing is done in hyperopt-sklearn)
# for more information see link in README file
def complexSpace(searchDict):
    # define your own complex space
    try:
        # combined
        # searchDict['rf__max_features'] = hp.choice('rf__max_features', [
        #     hp.choice('c1',
        #               ['sqrt', 'log2', None]),
        #     1 + hp.randint('c2',
        #                    3)
        # ])
        searchDict['rf__max_features'] = hp.choice('rf__max_features',
                                                   ['sqrt', 'log2',
                                                    None])
    except:
        print("not working")
    return searchDict
