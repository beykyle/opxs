import json

def parseParams(fname):
    with open(fname) as f:
        data = json.load(f)

    return data

