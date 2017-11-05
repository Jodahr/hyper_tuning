from sklearn.externals import joblib
import json
import pprint
from hyperopt import hp
import sys
import parameter_space as ps


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
        elif option['type'] == 'uniform':
            print(option['value'])
            space[parameter] = hp.uniform(parameter, option['value'][0],
                                          option['value'][1])
        elif option['type'] == 'randint':
            minimum = option['value']['min']
            maximum = option['value']['max']
            stepsize = option['value']['step']
            space[parameter] = minimum + (
                stepsize * hp.randint(parameter,
                                      (maximum - minimum) // stepsize + 1))
            print("hello")
        elif option['type'] == 'randrange':
            minimum = option['value']['min']
            maximum = option['value']['max']
            stepsize = option['value']['step']
            values = [x for x in range(minimum, maximum+1, stepsize)]
            space[parameter] = hp.choice(parameter, values)
            print("hello")
        else:
            print("option not recognized.", file=sys.stderr)
    # add some flag to use it
    #space = ps.complexSpace(space)
    return space
