from HOUDINI.Synthesizer import ReprUtils, ASTUtils
from HOUDINI.Synthesizer.AST import PPFuncApp, PPTermUnk, PPSortVar, PPInt, PPReal, PPVar, PPFuncSort, \
    PPTensorSort, PPDimConst, PPListSort
from HOUDINI.Synthesizer.ASTUtils import applyTd
from HOUDINI.Synthesizer.Unification import substOne


def _processProg(prog, cmts):
    newProgs = []
    sortVars = ASTUtils.getSortVars(prog)
    if sortVars:
        sortVar = sortVars[0]
        progress = True
        for cmt in cmts:
            newProg = substOne(sortVar, cmt, prog)
            newProgs.append(newProg)
    else:
        # Add the program as it is
        progress = False
        newProgs.append(prog)

    return newProgs, progress


def _processProgList(progs, cmts):
    progsNext = []
    progress = False
    for prog in progs:
        newProgs, iProgress = _processProg(prog, cmts)
        progress = progress or iProgress
        progsNext.extend(newProgs)

    return progsNext, progress


def instantiateSortVar(prog, cmts, maxSortVarsToBeInstantiated):
    resProgs = [prog]
    progress = True
    i = 0
    while progress and i < maxSortVarsToBeInstantiated:
        resProgs, progress = _processProgList(resProgs, cmts)
        i += 1

    return resProgs

def main():
    # term = PPVar(name='x')
    # print(term)
    # print(doubleVarNames(term))
    # exit()
    # for term in TermsRepo.termsrepo:
    #     print('############')
    #     print(term)
    #     print(doubleVarNames(term))

    prog1 = PPFuncApp(
        fn=PPVar(name='lib.compose'),
        args=[
            PPTermUnk(name='nn_fun_csc4_2',
                      sort=PPFuncSort(
                          args=[
                              PPSortVar(name='B')],
                          rtpe=PPTensorSort(param_sort=PPReal(), shape=[PPDimConst(value=1), PPDimConst(value=1)]))),
            PPTermUnk(name='nn_fun_csc4_3',
                      sort=PPFuncSort(
                          args=[PPListSort(param_sort=PPTensorSort(param_sort=PPReal(),
                                                                   shape=[PPDimConst(value=1), PPDimConst(value=1),
                                                                          PPDimConst(value=28),
                                                                          PPDimConst(value=28)]))],
                          rtpe=PPSortVar(name='B')))])

    prog2 = PPFuncApp(
        fn=PPVar(name='lib.compose'),
        args=[
            PPTermUnk(name='nn_fun_csc4_4',
                      sort=PPFuncSort(
                          args=[
                              PPSortVar(name='C')],
                          rtpe=PPTensorSort(param_sort=PPReal(), shape=[PPDimConst(value=1), PPDimConst(value=1)]))),
            PPTermUnk(name='nn_fun_csc4_5',
                      sort=PPFuncSort(
                          args=[PPListSort(param_sort=PPTensorSort(param_sort=PPReal(),
                                                                   shape=[PPDimConst(value=1), PPDimConst(value=1),
                                                                          PPDimConst(value=28),
                                                                          PPDimConst(value=28)]))],
                          rtpe=PPSortVar(name='C')))])

    prog3 = PPFuncApp(
        fn=PPVar(name='lib.compose'),
        args=[prog1, prog2])

    prog4 = PPFuncApp(fn=PPVar(name='lib.compose'), args=[PPTermUnk(name='nn_fun_csc4_8', sort=PPFuncSort(
        args=[PPListSort(param_sort=PPSortVar(name='B_1'))],
        rtpe=PPTensorSort(param_sort=PPReal(), shape=[PPDimConst(value=1), PPDimConst(value=1)]))),
                                                          PPFuncApp(fn=PPVar(name='lib.map_l'), args=[
                                                              PPTermUnk(name='nn_fun_csc4_9', sort=PPFuncSort(args=[
                                                                  PPTensorSort(param_sort=PPReal(),
                                                                               shape=[PPDimConst(value=1),
                                                                                      PPDimConst(value=1),
                                                                                      PPDimConst(value=28),
                                                                                      PPDimConst(value=28)])],
                                                                                                              rtpe=PPSortVar(
                                                                                                                  name='B_1')))])])
    cmts = [PPInt(), PPReal()]
    eprogs = instantiateSortVar(prog4, cmts)
    for i, eprog in enumerate(eprogs):
        print(i, ReprUtils.repr_py(eprog))
        print(i, eprog)


if __name__ == '__main__':
    main()
