#! /usr/bin/env python
import sys
import utils
import configparser
import json
import params
import search

#from geographiclib.geodesic import Geodesic
#import math
#geod = Geodesic.WGS84

def main():
    # Welcome Banner
    utils.printBanner()

    # Loading ConfigIni
    config = configparser.ConfigParser()
    config.read("../config/hyper_tuning.ini")
    config_pathsDict = utils.configSectionMap(config, "Paths")
    config_dataDict = utils.configSectionMap(config, "Data")
    config_modulesList = json.loads(
        utils.configSectionMap(config, "Modules")["importlist"])

    # dynamic module import and namespace update
    moduleNames = utils.loadModules(config_modulesList)
    globals().update(moduleNames)

    # get parameterSpace
    print(config_pathsDict)
    paramsFile = config_pathsDict["modelparams_json"]
    space = params.getParameterSpace(paramsFile)
    print(space)

    # get data
    dataPath = config_pathsDict["data"]
    label = config_dataDict["label"]
    data = utils.getData(dataPath, label)
    print(data)

    # get model
    modelPath = config_pathsDict["model"]
    model = utils.loadObject(modelPath)
    utils.printModelParams(model)

    # create Search
    opt = search.Search(model, data, space, outputFolder="bla")
    opt.run()

    # GoodBye Statement
    print('Thanks for using HyperTuning.\n')


# if executed as a script
# code will not be executed if module is imported
if __name__ == '__main__':
    # returns the return value of the main function to the system
    sys.exit(main())
