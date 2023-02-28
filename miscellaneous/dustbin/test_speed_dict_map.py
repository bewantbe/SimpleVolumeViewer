#!/usr/bin/env python3

import numpy as np

n = int(1e6)

sid = 1*np.argsort(np.random.rand(n))

n_m = np.max(sid) + 1
amap = np.zeros(n_m, dtype=np.int32)
amap[sid] = np.arange(n)
# 3.59 ms ± 84.2 µs per loop
#x10: 10.3 ms ± 309 µs

dmap = dict(zip(sid, np.arange(n)))
# 132 ms ± 4.95 ms per loop

dmap = dict(zip(sid, range(n)))
# 93.4 ms ± 862 µs per loop

dmap = dict(np.stack((sid, np.arange(n)), axis=1))
# 710 ms ± 9.91 ms per loop

v = zip(sid, np.arange(n))
dmap = dict(v)
# error

v = np.stack((sid, np.arange(n)), axis=1)
# 2.65 ms ± 13.4 µs
dmap = dict(v)
# 702 ms ± 7.71 ms per loop

# tuples
v = [(a,b) for a,b in zip(sid, np.arange(n))]
# 112 ms ± 3.14 ms per loop
dmap = dict(v)
# 103 ms ± 8 ms per loop

# c++
# ~0.034 s

def fn():
    s = 0
    for i in range(n):
        s += i
    return s
# 26.3 ms ± 521 µs per loop

def fs():
    s = 0
    for i in range(n):
        s += sid[i]
    return s
# 70.5 ms ± 1.24 ms per loop

def fm():
    s = 0
    for i in range(n):
        s += dmap[sid[i]]
    return s
#x10 168 ms ± 9.23 ms per loop
#    140 ms ± 10.3 ms per loop

def fa():
    s = 0
    for i in range(n):
        s += amap[sid[i]]
    return s
#x10 249 ms ± 5.54 ms per loop
#    186 ms ± 7.82 ms per loop

x = amap[sid]
# 1.18 ms ± 35.3 µs per loop

x = [dmap[i] for i in sid]
# 78.7 ms ± 481 µs per loop

x = np.array([dmap[i] for i in sid])
# 107 ms ± 1.43 ms per loop

x = np.array([dmap[i] for i in sid], dtype=np.int32)
# 123 ms ± 1.87 ms per loop

x = np.vectorize(dmap.get)(sid)
# 85.1 ms ± 2.08 ms per loop

x = np.vectorize(dmap.__getitem__)(sid)
# 83.1 ms ± 435 µs per loop

x = np.vectorize(dmap.__getitem__)(sid).astype(np.int32)
# 83.4 ms ± 530 µs per loop

x = np.array(list(map(dmap.get, sid)))
# 90.8 ms ± 644 µs per loop

# C++
# ~0.008 s

"""
Summary:

 Initilize np.array is >10 times faster than dict().
 But np.array is 1.3~1.5 times (overall) slower than dict() for indexing in for loop.

"""

