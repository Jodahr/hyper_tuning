#! /usr/bin/env python
import sys
import argparse
import configparser
import json
from importlib import import_module
import lib.utils as ut
import search


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
    print(config.values())
    #configDict = {}
    #configDict['dataPath'] = configSectionMap(config, "Paths")['data']
    #configDict['modelPath'] = configSectionMap(config, "Paths")['model']
    #configDict['dataType'] = configSectionMap(config, "Data")['type']
    #configDict['dataLabel'] = configSectionMap(config, "Data")['label']
    #configDict['dataFormat'] = configSectionMap(config, "Data")['format']
    #configDict['classification'] = configSectionMap(config,
    #                                                "Data")['classification']
    #return configDict

def importModules(importList):
    for element in importList:
        if not element['alias']:
            globals()[element['name']] = import_module(name=element['name'])
        else:
            globals()[element['alias']] = import_module(name=element['name'])


def main():
    test2 = search.Search(model='bla', paramSpace='blubb', data='bla')
    fmin = search.Fmin(model='bla',X='x', y='y', scoring='roc_auc', shuffe=False)
    fmin.__fmin__()
    #test2.objective()
    utils = ut.Utils()
    utils.parse()
    config = configparser.ConfigParser()
    config.read("../config/hyper_tuning.ini")
    test = json.loads(config.get("Modules", "importList"))
    importModules(test)
    print(test)
    getConfigs()
    print('End\n')


# if executed as a script
# code will not be executed if module is imported
if __name__ == '__main__':
    # returns the return value of the main function to the system
    sys.exit(main())
