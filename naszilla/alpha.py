import numpy as np

c21 = 1
c11 = 1
r1 = 8
c22 = 1
c12 = 10
r2 = 4

r = 200
x = 4.252

def dist(x):
    return np.abs(np.sqrt((c21-c11-r1*np.cos(x)) ** 2 + (c22-c12-r1*np.sin(x)) ** 2) - r2)

if __name__ == '__main__':
    for c in [1.1, 1.01, 1.001, 1.0001, 1.00001]:
        print(f'c = {c}; ratio = {dist(c * x)/((c ** r)*dist(x))}')