from HOUDINI.Synthesizer.AST import *
from HOUDINI.Synthesizer.ASTDSL import mkFuncSort, mkListSort
from HOUDINI.Synthesizer.ReprUtils import simplerep
from HOUDINI.Synthesizer.Unification import unifyLists, applySubst


def isnamedtupleinstance(x):
    t = type(x)
    b = t.__bases__
    if not isinstance(x, tuple): return False
    f = getattr(t, '_fields', None)
    if not isinstance(f, tuple): return False
    return all(type(n) == str for n in f)


def ntstr(x):
    if isnamedtupleinstance(x):
        fields = ','.join([ntstr(getattr(x, k)) for k in x._fields])
        return x.__class__.__name__ + '(%s)' % fields
    elif isinstance(x, tuple):
        return "(%s)" % ', '.join([ntstr(e) for e in x])
    elif isinstance(x, list):
        return "[%s]" % ', '.join([ntstr(e) for e in x])
    else:
        return str(x)


def mkTable(rows):
    res = ''
    res += '<table  border="1">'
    for row in rows:
        res += '<tr>'
        for data in row:
            res += '<td> %s </td>' % data
        res += '</tr>'
    res += '</table>'
    return res


def displayUnifyList(xs, ys):
    subst = unifyLists(xs, ys)

    rows = []
    for x, y in zip(xs, ys):
        row = [x, y, applySubst(subst, x), applySubst(subst, y)]
        row = [simplerep(d) for d in row]
        rows.append(row)

    tab1 = mkTable(rows)

    rows = []
    for p, q in subst:
        row = [p, q]
        row = [simplerep(d) for d in row]
        rows.append(row)
    tab2 = mkTable(rows)

    return tab1 + tab2


def displayUnifyListH(xs, ys):
    subst = unifyLists(xs, ys)

    row1 = ['xs'] + list(map(simplerep, xs))
    row2 = ['ys'] + list(map(simplerep, ys))
    row3 = ['xs+'] + list(map(simplerep, applySubst(subst, xs)))
    row4 = ['ys+'] + list(map(simplerep, applySubst(subst, ys)))

    tab1 = mkTable([row1, row2, row3, row4])

    rows = []
    for p, q in subst:
        row = [p, q]
        row = [simplerep(d) for d in row]
        rows.append(row)
    tab2 = mkTable(rows)

    return tab1 + '<br>' + tab2


def main():
    try:
        # s1 = mkIntTensorSort(['A', 3])
        # print(ntstr(s1))
        #
        # t1 = mkIntTensorSort([2, 'B'])
        # print(ntstr(t1))
        #
        # subst = unifyLists([s1], [t1])
        # for x, y in subst:
        #     print("%s ---> %s" % (ntstr(x), ntstr(y)))

        t1 = PPSortVar('T1')
        t2 = PPSortVar('T2')
        s = PPSortVar('S')
        x = mkFuncSort(mkFuncSort(t1, t2), mkListSort(t1), mkListSort(t2))
        y = mkFuncSort(mkFuncSort(s, s), mkListSort(s), mkListSort(s))
        displayUnifyListH([x], [y])

    except:
        print("exception")
        raise


if __name__ == '__main__':
    main()
