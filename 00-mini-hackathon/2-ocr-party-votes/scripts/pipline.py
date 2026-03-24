from functools import reduce

def pipe(value, *funcs):
    return reduce(lambda v, f: f(v), funcs, value)