"""Store several named selections and choose which to apply"""

import functools
import operator


class SelectionSet:
    def __init__(self, aliases = {}, **selections):
        self._selections = selections
        self._aliases = aliases

    
    def __getattr__(self, name):
        if name in self._aliases:
            return functools.reduce(operator.and_, (self._selections[n] for n in self._alises[name]))
        elif name in self._selections:
            return self._selections[name]
        else:
            return super().__getattr__(self, name)


    def __call__(self, *names):
        return functools.reduce(
                operator.and_,
                (getattr(self, name) for name in names)
        )
