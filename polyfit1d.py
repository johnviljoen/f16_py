import torch
import numpy as np

# 5 points
p0 = (0,1)
p1 = (1,3)
p2 = (2,4)
p3 = (3,9)
p4 = (4,10)

dtype = torch.float32

# 4th order approximation
X = torch.tensor([  
    [p0[1]**0,p0[1]**1,p0[1]**2,p0[1]**3,p0[1]**4],
    [p1[1]**0,p1[1]**1,p2[1]**2,p2[1]**3,p2[1]**4],
    [p2[1]**0,p2[1]**1,p2[1]**2,p2[1]**3,p2[1]**4],
    [p3[1]**0,p3[1]**1,p3[1]**2,p3[1]**3,p3[1]**4],
    [p4[1]**0,p4[1]**1,p4[1]**2,p4[1]**3,p4[1]**4]
], dtype=dtype)

Y = torch.tensor([
    [p0[0]],
    [p1[0]],
    [p2[0]],
    [p3[0]],
    [p4[0]]
], dtype=dtype)

b = torch.linalg.inv(X.T @ X) @ X.T @ Y

p = (4,10)
p = (3,9)

x = torch.tensor([[p[1]**0,p[1]**1,p[1]**2,p[1]**3,p[1]**4]], dtype=dtype)

import ipdb
ipdb.set_trace()