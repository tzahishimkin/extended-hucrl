import sys
from contextlib import ExitStack
from itertools import product
import argparse
import sys
import itertools


def get_permutations():
    parser = argparse.ArgumentParser()
    argv = sys.argv[1:]
    assert argv[0][0:2] == '--', 'requires -- in the first work'
    res = [i for i, st in enumerate(argv) if '--' in st] + [len(argv)]
    d = {}
    for ires in range(len(res) - 1):
        d[argv[res[ires]]] = [argv[i] for i in range(res[ires] + 1, res[ires + 1])]

    flags = ''
    for key, value in d.copy().items():
        if value == []:
            flags += ' ' + key
            del d[key]

    combinations = list(itertools.product(*(d[Name] for Name in d.keys())))

    def get_str_from_lists(keys, values):
        AAA = [[key, value] for key, value in zip(keys, values)]
        merged = list(itertools.chain(*AAA))
        merged = ' '.join(merged)
        # print(merged)
        return merged

    get_str_from_lists(keys=d.keys(), values=combinations[0])
    list_args = [get_str_from_lists(keys=d.keys(), values=comb) for comb in combinations]
    list_args = [flags + ' ' + list_arg for list_arg in list_args]
    return list_args


if __name__ == '__main__':
    list = get_permutations()
    for l in list:
        print(l)
