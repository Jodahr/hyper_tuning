import argparse
import configparser
import json
import logging
import sys


class Utils():
    def parse(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-m', '--modelname', default=None, type=str,
                            help='a name for your model')
        args = parser.parse_args()
        print(args.modelname)
        #print("hello")
        return args
