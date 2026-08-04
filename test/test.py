import mlx.core as mx

a = mx.array([1,2,3,4,5])
b = mx.array([1,2,])
print(b.all() in a.all())
