from collections import defaultdict, Counter
import json

from matplotlib import pyplot as plt
import numpy as np
import torch

def print_encoding(model_inputs,indent=4):
    indent_str = " "*indent
    print("{")
    for k,v in model_inputs.items():
        print(indent_str+k+":")
        print(indent_str+indent_str+str(v))
    print("}")

print_encoding({"axa":"abcd"},indent=0)
        
    