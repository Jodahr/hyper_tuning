from sklearn.externals import joblib
import json
import pprint
from hyperopt import hp
import sys


def loadModel(modelpath):
    model = joblib.load(modelpath)
    return model
    

def printModelParams(model):
    modelDict = model.get_params()
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(modelDict)
#    print(json.dumps(str(modelDict), indent=2))


def parameterSpace(paramsFile):
    space = {}
    with open(paramsFile) as data_file:
        paramsDict = json.load(data_file)
    for parameter, option in paramsDict.items():
        if option['type'] == 'choice':
            print(option['value'])
            space[parameter] = hp.choice(parameter, option['value'])
            #space['rf__criterion'] = hp.choice('rf__criterion', ['gini', 'entropy'])
        elif option['type'] == 'uniform':
            print(option['value'])
            #space[parameter] = hp.uniform(parameter, option['value'][0],
            #                              option['value'][1])
        elif option['type'] == 'randint':
            space[parameter] = 1 + hp.randint(parameter,
                                          option['value'][1])
            #space[parameter] = 2 + hp.randint(parameter,
            #                              5)
            print("hello")
        else:
            print("option not recognized.", file=sys.stderr)
    return space
