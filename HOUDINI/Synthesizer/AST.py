from typing import NamedTuple, Union, List


class PPSymbol(NamedTuple('PPSymbol', [('value', str)])):
    pass


# Custom sorts
class PPImageSort(NamedTuple('PPImageSort')):
    pass


# PPSort
class PPInt(NamedTuple('PPInt')):
    pass


class PPReal(NamedTuple('PPReal')):
    pass


class PPBool(NamedTuple('PPBool')):
    pass


class PPFuncSort(NamedTuple('PPFuncSort', [('args', List['PPSort']),
                                           ('rtpe', 'PPSort')])):
    pass


class PPListSort(NamedTuple('PPListSort', [('param_sort', 'PPSort')])):
    pass


class PPGraphSort(NamedTuple('PPGraphSort', [('param_sort', 'PPSort')])):
    pass


class PPTensorSort(NamedTuple('PPTensorSort', [('param_sort', 'PPSort'), ('shape', List['PPDim'])])):
    pass


class PPSortVar(NamedTuple('PPSortVar', [('name', str)])):
    pass


class PPEnumSort(NamedTuple('PPEnumSort', [('start', int), ('end', int)])):
    """
    Useful as a meta variable. (Example: integer parameter for the "repeat" function)
    """
    pass


class PPDimVar(NamedTuple('PPDimVar', [('name', str)])):
    pass


class PPDimConst(NamedTuple('PPDimConst', [('value', int)])):
    pass


PPDim = Union[PPDimVar, PPDimConst]
PPDimTypes = (PPDimVar, PPDimConst)

PPSort = Union[PPInt, PPReal, PPBool, PPSortVar, PPListSort, PPGraphSort, PPTensorSort, PPFuncSort, PPImageSort,
               PPEnumSort]
PPSortTypes = (PPInt, PPReal, PPBool, PPSortVar, PPListSort, PPGraphSort, PPTensorSort, PPFuncSort, PPImageSort,
               PPEnumSort)


class PPIntConst(NamedTuple('PPIntConst', [('value', int)])):
    pass


class PPRealConst(NamedTuple('PPRealConst', [('value', float)])):
    pass


class PPBoolConst(NamedTuple('PPBoolConst', [('value', bool)])):
    pass


class PPVar(NamedTuple('PPVar', [('name', str)])):
    pass


LambdaBase = NamedTuple('LambdaBase', [('params', List['PPVarDecl']),
                                       ('body', 'PPTerm')])


class PPLambda(LambdaBase):
    def __new__(cls, *args, **kwargs):
        return LambdaBase.__new__(cls, *args, **kwargs)


# class PPLambda(NamedTuple('PPLambda', [('params', List['PPVarDecl']),
#                                        ('body', 'PPTerm')])):
#     pass


# class PPFuncApp(NamedTuple('PPFuncApp', [('fname', PPSymbol),
#                                          ('args', List['PPTerm'])])):
#     pass

class PPFuncApp(NamedTuple('PPFuncApp', [('fn', 'PPTerm'),
                                         ('args', List['PPTerm'])])):
    pass


class PPListTerm(NamedTuple('PPListTerm', [('items', List['PPTerm'])])):
    pass


class PPTermNT(NamedTuple('PPTermNT', [('name', str), ('sort', PPSort)])):
    pass


class PPTermUnk(NamedTuple('PPTermUnk', [('name', str), ('sort', PPSort)])):
    pass


PPTerm = Union[PPIntConst, PPRealConst, PPBoolConst, PPVar, PPLambda, PPFuncApp, PPListTerm,
               PPTermNT, PPTermUnk]
PPTermTypes = (PPIntConst, PPRealConst, PPBoolConst, PPVar, PPLambda, PPFuncApp, PPListTerm,
               PPTermNT, PPTermUnk)

PPNT = Union[PPTermNT]
PPNTTypes = (PPTermNT)


class PPVarDecl(NamedTuple('PPVarDecl', [('name', str), ('sort', PPSort)])):
    pass


class PPFunc(NamedTuple('PPFunc', [('fname', PPSymbol)])):
    pass


class PPFuncDecl(NamedTuple('PPFuncDecl', [('fname', PPSymbol),
                                           ('sort', PPFuncSort)])):
    pass


class PPFuncDef(NamedTuple('PPFuncDef', [('fname', PPSymbol),
                                         ('params', List[PPVar]),
                                         ('body', PPTerm),
                                         ('rsort', PPSort)])):
    pass


if __name__ == '__main__':
    pass
