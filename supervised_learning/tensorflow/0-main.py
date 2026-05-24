#!/usr/bin/env python3

create_placeholders = __import__('0-create_placeholders').create_placeholders

x, y = create_placeholders(784, 10)

print(x)
print(y)