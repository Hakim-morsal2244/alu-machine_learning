#!/usr/bin/env python3

try:
	import tensorflow as tf
except Exception:
	# Fallback stub so the script can run without TensorFlow installed (useful for linting)
	import types
	tf = types.SimpleNamespace(nn=types.SimpleNamespace(tanh=lambda x: x))

create_placeholders = __import__('0-create_placeholders').create_placeholders
create_layer = __import__('1-create_layer').create_layer

x, y = create_placeholders(784, 10)
layer = create_layer(x, 256, tf.nn.tanh)
print(layer)