from Optimization.Functions import pathtime as pt
from Optimization.Algorithm import classy, optisolve as op
from Optimization.Data import project5data as p5d
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Project 5
N = 12  # This is the order of the fit (2N decision variables)
# read in the velocity data array defined on
# [0,1]x[0,1] and set the path end points
v = pd.read_csv('../Data/SpeedData.csv', header=None).to_numpy()
# v = p5d.data(30)
my, mx = v.shape
A = (.05, .05)
B = (.95, .95)

para = classy.para(0.0001, 0.19, 0, [v, A, B], 0, 0, 0)
pr = classy.funct(pt.pathtime, 'LBFGS', 'strongwolfe', 0.2 * np.random.randn(2 * N), para, 1)

res = op.optimize(pr)

# plot the optimal path superimposed on the velocity image
smp = 1000  # number of points defining the path
FigDPI = 256  # figure dpi (effects scale)
FigSize = (8, 6)  # figure size
ColorMap = 'jet'  # velocity colormap
LineColor = 'white'  # path plot color
LineWidth = 1  # path line width
PointSize = 16  # size of path endpoints

r = np.linspace(0, 1, smp)
xx = (1 - r) * A[0] + r * B[0]
yy = (1 - r) * A[1] + r * B[1]
for k in range(N):
    s = np.sin((k + 1) * np.pi * r)
    xx += res.input[k] * s
    yy += res.input[k + N] * s
xxr = xx * (mx - 1)
yyr = yy * (my - 1)
fig = plt.figure(dpi=FigDPI, figsize=FigSize)
ax = fig.add_subplot()
vim = ax.imshow(v, cmap=ColorMap)
plt.colorbar(vim, orientation='vertical')
ax.plot(yyr, xxr, color=LineColor, linewidth=LineWidth)
ax.scatter(A[1] * mx, A[0] * my, PointSize, LineColor)
ax.scatter(B[1] * mx, B[0] * my, PointSize, LineColor)
plt.xticks([])
plt.yticks([])
plt.show()
