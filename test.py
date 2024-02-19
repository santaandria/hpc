import sys
import numpy as np

print(f"Python interpreter location: {sys.executable}")
n = 5000
A = np.random.rand(n,n)
B = np.random.rand(n,n)
C = np.matmul(A,B)
print('Done')
