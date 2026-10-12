import itertools
from collections.abc import Mapping, MutableMapping
from typing import cast

import mergedeep


def merge[M: MutableMapping](destination: M, *sources: Mapping) -> M:
    """Deep-merge sources into destination, as mergedeep.merge, and return destination.

    mergedeep annotates its return as any MutableMapping, though it returns the
    destination it was given; this keeps the destination's type.
    """
    return cast(M, mergedeep.merge(destination, *sources))


def interleave(*a):
    return tuple(itertools.chain.from_iterable(tuple(zip(*a, strict=False))))


def tile(a, n):
    return tuple(itertools.chain.from_iterable(itertools.repeat(a, n)))


def repeat(a, n):
    return tuple(interleave(*itertools.repeat(a, n)))
