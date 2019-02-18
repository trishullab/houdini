import errno
import os
import random

from HOUDINI.Synthesizer.AST import *
import os


def genFuncSort():
    m = 3
    numArgs = random.randint(0, m)
    argSorts = [genSort() for i in range(numArgs)]
    rtpe = genSort()
    return PPFuncSort(argSorts, rtpe)


def genSort():
    sorts = [PPInt, PPReal, PPBool, PPFuncSort, PPListSort, PPGraphSort, PPTensorSort]
    theSort = random.choice(sorts)
    if theSort == PPInt:
        return PPInt()
    elif theSort == PPReal:
        return PPReal()
    elif theSort == PPBool:
        return PPBool()
    elif theSort == PPFuncSort:
        return genFuncSort()
    elif theSort == PPListSort:
        return genPPListSort()
    elif theSort == PPGraphSort:
        return genPPGraphSort()
    elif theSort == PPTensorSort:
        return genTensorSort()


def genPPListSort():
    sort = genSort()
    return PPListSort(sort)


def genPPGraphSort():
    sort = genSort()
    return PPGraphSort(sort)


def genTensorSort():
    maxNumDims = 5
    maxDimValue = 5000
    sort = genSort()
    shape: List[PPDim] = None
    numDims = random.randint(0, maxNumDims)
    shape = [PPDimConst(random.randint(0, maxDimValue)) for i in range(numDims)]
    return PPTensorSort(sort, shape)


def createDir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def getPythonPath():
    try:
        user_paths = os.environ['PYTHONPATH'].split(os.pathsep)
    except KeyError:
        user_paths = []
    return user_paths


def getPath():
    try:
        user_paths = os.environ['PATH'].split(os.pathsep)
    except KeyError:
        user_paths = []
    return user_paths
