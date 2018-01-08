import argparse
from importlib import import_module
import pprint
import dill
import pandas as pd

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', default=None, type=str,
                        help='Path to output folder.')
    parser.add_argument('-p', '--params', default=None, type=str,
                        help='Path to parameter file.')
    parser.add_argument('-i', '--iterations', default=None, type=int,
                        help='Number of iterations')
    parser.add_argument('-s', '--steps', default=None, type=int,
                        help='Number of iterations until model gets saved.')
    parser.add_argument('-s', '--steps', default=None, type=int,
                        help='Number of iterations until model gets saved.')
    args = parser.parse_args()
    return args


def configSectionMap(config, section):
    configDict = {}
    options = config.options(section)
    for option in options:
        configDict[option] = config.get(section, option)
    return configDict


def loadModules(moduleList):
    globals()["name"] = "Marcel"
    for mod in moduleList:
        print("load module {}...".format(mod["name"]))
        if not mod['alias'] and not mod['package']:
            globals()[mod['name']] = import_module(name=mod['name'],
                                                   package=mod['package'])
        if not mod['alias']:
            globals()[mod['package']] = import_module(name=mod['name'],
                                                   package=mod['package'])
        else:
            globals()[mod['alias']] = import_module(name=mod['name'],
                                                    package=mod['package'])
        print("...done\n")
    return globals()


def printModelParams(model):
    modelDict = model.get_params()
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(modelDict)


def loadObject(objectPath):
    # with statement do not create new exec scope
    with open(objectPath, 'rb') as f:
        obj = dill.load(f)
    return obj


def saveObject(objectPath):
    with open(objectPath, 'wb') as f:
        dill.dump(f)


def printBanner():
    print("Welcome to HyperTuning.\n")


def getData(dataPath, label):
    data = dill.load(open(dataPath, 'rb'))
    y = data[label]
    X = data.drop(label, axis=1)
    return (X, y)
