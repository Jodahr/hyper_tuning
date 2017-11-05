#! /usr/bin/env python
import sys
import argparse
import configparser
import logging
import params as pm
import data as d
import optimize as om
from sklearn.preprocessing import LabelBinarizer
import numpy as np
from geographiclib.geodesic import Geodesic
import math
geod = Geodesic.WGS84


# parser (still needs to be finalized)
def parse():
    parser = argparse.ArgumentParser(description='my parser')
    parser.add_argument('task', type=str, help='choose between tasks')
    parser.add_argument('-m', '--modelname', default=None, type=str,
                        help='a name for your model')
    args = parser.parse_args()
    return args


def configSectionMap(config, section):
    dict1 = {}
    options = config.options(section)
    for option in options:
        try:
            dict1[option] = config.get(section, option)
            if dict1[option] == -1:
                logging.debug("skip: %s" % option)
        except:
            logging.debug("exception on %s!" % option)
            dict1[option] = None
    return dict1


def getConfigs():
    config = configparser.ConfigParser()
    config.read("../config/hyper_tuning.ini")
    print(config.sections())
    configDict = {}
    configDict['dataPath'] = configSectionMap(config, "Paths")['data']
    configDict['modelPath'] = configSectionMap(config, "Paths")['model']
    configDict['dataType'] = configSectionMap(config, "Data")['type']
    configDict['dataLabel'] = configSectionMap(config, "Data")['label']
    configDict['dataFormat'] = configSectionMap(config, "Data")['format']
    configDict['classification'] = configSectionMap(config,
                                                    "Data")['classification']
    return configDict


# all executable code goes here
def main():
    #parse()
    configDict = getConfigs()
    modelPath = configDict['modelPath']
    model = pm.loadModel(modelPath)
    dataPath = configDict['dataPath']
    label = configDict['dataLabel']
    dataFormat = configDict['dataFormat']
    classification = configDict['classification']
    data = d.getData(dataPath, label)
    # print(data['train'][0])
    # pm.printModelParams(model)
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logging.debug('This is a log message.')
    space = pm.parameterSpace("../input/parameter_space.json")
    X = data['train'][0]
    y = data['train'][1]
    X_test = data['test'][0]
    y_test = data['test'][1]
    if classification == 'multiclass':
        y = LabelBinarizer().fit_transform(y)
        y_test = LabelBinarizer().fit_transform(y_test)
    y = y.apply(lambda x: 0 if x == 2 else x)  # for binary classification
    y_test = y_test.apply(lambda x: 0 if x == 2 else x)  # for binary classification
    search = om.Search(model, space, X, y, X_test, y_test)
    #search.run()
    search.inf_search()
    #print(search.space)
    print('End\n')


# if executed as a script
# code will not be executed if module is imported
if __name__ == '__main__':
    # returns the return value of the main function to the system
    sys.exit(main())
