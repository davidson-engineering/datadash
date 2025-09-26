import itertools


def interleave(*a):
    return tuple(itertools.chain.from_iterable(tuple(zip(*a))))


def tile(a, n):
    return tuple(itertools.chain.from_iterable(itertools.repeat(a, n)))


def repeat(a, n):
    return tuple(interleave(*itertools.repeat(a, n)))
