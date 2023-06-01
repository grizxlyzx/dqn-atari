import zlib
import numpy as np
import torch
import itertools
import objgraph
from collections import deque

if __name__ == '__main__':



    objgraph.show_growth()
    d = deque(maxlen=100)
    for i in range(12):
        d.append(1)
    print('---')
    objgraph.show_growth()
