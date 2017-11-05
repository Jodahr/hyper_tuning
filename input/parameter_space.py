from hyperopt import hp


# for more complex search spaces
def complexSpace(searchDict):
    # define your own complex space
    try:
        # combined
        searchDict['rf__max_features'] = hp.choice('rf__max_features', [
            hp.choice('c1',
                      ['sqrt', 'log2']),
            1 + hp.randint('c2',
                           3)
        ])
    except:
        print("not working")
    return searchDict
