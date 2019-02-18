from typing import Dict, Optional
import torch.nn as nn
import pickle
import os.path

from HOUDINI.Synthesizer.AST import *
from HOUDINI.Synthesizer.AST import PPSort
from HOUDINI.Synthesizer.GenUtils import createDir

"""
t = PPSortVar('T')
t1 = PPSortVar('T1')
t2 = PPSortVar('T2')
"""


class PPLibItem(NamedTuple('PPLibItem', [('name', str), ('sort', PPSort), ('obj', object)])):
    pass


class FnLibrary:
    def __init__(self):
        self.items: Dict[str, PPLibItem] = {}
        # self._add_python_fns()

    def save(self, location):
        """
        Saves the neural networks if already not on disk.
        Saves the name-sort dictionary to lib.pickle
        """

        def isNN(anObj):
            return issubclass(type(anObj), nn.Module)

        if not os.path.exists(location):
            createDir(location)

        for name, li in self.items.items():
            if isNN(li.obj):
                nnFileName = location + '/' + name + '.pth'
                if not os.path.isfile(nnFileName):
                    createDir(location)
                    li.obj.save(location)
            else:
                pass

        newDict = {}
        for name, li in self.items.items():
            newDict[name] = (li.sort, isNN(li.obj))

        with open(location + '/lib.pickle', 'wb') as fh:
            pickle.dump(newDict, fh, protocol=pickle.HIGHEST_PROTOCOL)

    def save1(self, location, id):
        """
        Saves the neural networks if already not on disk.
        Saves the name-sort dictionary to lib.pickle
        """

        def isNN(anObj):
            return issubclass(type(anObj), nn.Module)

        for name, li in self.items.items():
            if isNN(li.obj):
                nnFileName = location + '/' + name + '.pth'
                if not os.path.isfile(nnFileName):
                    createDir(location)
                    li.obj.save(location)
            else:
                pass

        newDict = {}
        for name, li in self.items.items():
            newDict[name] = (li.sort, isNN(li.obj))

        with open(location + '/lib{}.pickle'.format(id), 'wb') as fh:
            pickle.dump(newDict, fh, protocol=pickle.HIGHEST_PROTOCOL)

    def addItem(self, libItem: PPLibItem):
        self.items[libItem.name] = libItem
        self.__dict__[libItem.name] = libItem.obj

    def addItems(self, libItems: List[PPLibItem]):
        for li in libItems:
            self.addItem(li)

    def removeItems(self, names: List[str]):
        for name in names:
            self.items.pop(name, None)
            self.__dict__.pop(name, None)

    def get(self, name: str) -> Optional[PPLibItem]:
        res = None
        if name in self.items:
            res = self.items[name]
        return res

    def getWithLibPrefix(self, name: str) -> Optional[PPLibItem]:
        res = None
        if name in self.items:
            res = self.items[name]
            res = PPLibItem('lib.' + res.name, res.sort, res.obj)
        return res

    def set(self, name: str, obj: callable):
        if name in self.items:
            self.items[name].obj = obj

    def getDict(self) -> Dict[str, PPLibItem]:
        return self.items.copy()
