#!/usr/bin/env python3

import pickle

with open("27-saved.pkl", "rb") as f:
    obj = pickle.load(f)

print(type(obj))