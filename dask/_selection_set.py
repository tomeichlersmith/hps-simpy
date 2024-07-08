"""Store several named selections and choose which to apply"""

import functools
import operator


class SelectionSet:
    def __init__(self, aliases = {}, **selections):
        self._selections = selections
        self._aliases = aliases

    
    def __getattr__(self, name):
        if name in self._aliases:
            return functools.reduce(
                operator.and_,
                (self._selections[n] for n in self._aliases[name])
            )
        elif name in self._selections:
            return self._selections[name]
        elif name in self.__dict__:
            return self.__dict__[name]
        else:
            raise ValueError(f'SelectionSet object has no attribute {name}')


    def __call__(self, *names):
        return functools.reduce(
                operator.and_,
                (getattr(self, name) for name in names)
        )
