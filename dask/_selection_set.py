"""Store several named selections and choose which to apply"""

import functools
import operator


class SelectionSet:
    def __init__(self, **selections):
        self.__dict__ = selections


    def all(self):
        return self.__call__(*list(self.__dict__.keys()))


    def __call__(self, *names):
        return functools.reduce(
                operator.and_,
                (getattr(self, name) for name in names)
        )