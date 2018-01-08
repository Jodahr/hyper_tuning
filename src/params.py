import json
from hyperopt import hp
import numpy as np
import sys


def getParameterSpace(paramsFile):
    # empty dict holding the parameterSpace
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
        elif option['type'] == 'loguniform':
            minimum = option['value'][0]
            maximum = option['value'][1]
            space[parameter] = hp.loguniform(parameter,
                                             np.log10(minimum)*np.log(10),
                                             np.log10(maximum)*np.log(10))
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
    return space
