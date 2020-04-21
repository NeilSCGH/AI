import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

#Getting X and Y
data = np.loadtxt(fname = "data.txt")
X1=data[:,:1]
X2=data[:,1:2]
Y=data[:,2:]

fig = plt.figure()
ax = fig.gca(projection='3d')

ax.scatter3D(X1, X2, Y)

ax.set_xlabel('X1 Label')
ax.set_ylabel('X2 Label')
ax.set_zlabel('Y Label')
plt.show()