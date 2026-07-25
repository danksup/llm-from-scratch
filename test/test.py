import numpy as np
import mlx.core as mx
from typing import Any
import inspect

E = 4
top_k = 2
ba =[
  [  # Batch 0

    [  # Head 0
      [ 1,  2,  3,  4],
      [ 5,  6,  7,  8],
      [ 9, 10, 11, 12]
    ],

    [  # Head 1
      [13, 14, 15, 16],
      [17, 18, 19, 20],
      [21, 22, 23, 24]
    ]

  ],

  [  # Batch 1

    [  # Head 0
      [25, 26, 27, 28],
      [29, 30, 31, 32],
      [33, 34, 35, 36]
    ],

    [  # Head 1
      [37, 38, 39, 40],
      [41, 42, 43, 44],
      [45, 46, 47, 48]
    ]

  ]
]
a = mx.array(ba)
B,n_kv_heads,T,_ = a.shape
w = 2
n = a.shape[-1]
pad = [(0,0), (0,0),(w-1,0), (0,0)]
b = mx.pad(a, pad)
P = T + w - 1
shape = a.shape[0], a.shape[1], a.shape[2], w, a.shape[3]
stride = a.shape[-1] * n_kv_heads * P, a.shape[-1] * P, a.shape[-1], a.shape[-1], 1

c = mx.as_strided(b, shape=shape, strides= stride)
an = np.asarray(a)
itemsize = an.dtype.itemsize
np_stride = tuple(s * itemsize for s in stride)
z = np.lib.stride_tricks.as_strided(b, shape=shape, strides= np_stride)
# print(c)
# print(z)
# # print(b)
# # print("\n")
# # print(c)
# # print(c.shape)
# # print(stride)

# T = 3
# w = 2
# a = mx.arange(T)
# g = mx.repeat(a, w+1)
# print(g)

# a = mx.arange(w+1)
# d = mx.tile(a, T)
# print(d)

# a = mx.array(3)
# b = mx.array(False)
# print(a)
# print(b)
# print(a*b)
# print(a.dtype)
# g = mx.log(a.astype(mx.bfloat16))
# print(g.dtype)

# a = np.sqrt(a, dtype=np.longdouble)
# print(a)
# a = {"transformer_configs":{}}
# transformer = a["transformer_configs"]
# transformer["vocab_size"] = 2
# print(a)

d = mx.random.uniform(-0.1, 0.1, (4096,4096), dtype=mx.float32)
i = np.float16

import time

a = time.perf_counter()
mx.eval(d)
b = time.perf_counter()
print(b-a)

a = time.perf_counter()
d = d.astype(mx.float32)
mx.eval(d)
b = time.perf_counter()
print(b-a)

a = time.perf_counter()
d = d.astype(mx.float16)
mx.eval(d)
b = time.perf_counter()
print(b-a)





