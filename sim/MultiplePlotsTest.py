import matplotlib.pyplot as plt
import math
import numpy as np

# Some example data to display
x = np.linspace(0, 2 * np.pi, 400)  # x=0<x<2pi, 400 steps
y = np.sin(x ** 2)  # y=sin(x^2)
a = 1/x


# fig, ax = plt.subplots()
# ax.plot(x, y)
# ax.set_title('A single plot')
# plt.show()


fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(x, y)
axs[0, 0].set_title('Axis [0, 0]')
axs[0, 1].plot(x, a, 'tab:orange')
axs[0, 1].set_title('Axis [0, 1]')
axs[1, 0].plot(x, -y, 'tab:green')
axs[1, 0].set_title('Axis [1, 0]')
axs[1, 1].plot(x, -a, 'tab:red')
axs[1, 1].set_title('Axis [1, 1]')

for ax in axs.flat:
    ax.set(xlabel='x-label', ylabel='y-label')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()

plt.show()


# x = np.array([
#     [0.],
#     [0.],
#     [0.],
#     [0.]])

# x_d = np.array([
#     [0.],
#     [0, 1.0, 2.0 3, 4],
#     [0.],
#     [0.]])
# # First find unit vector form robot in direction of heading
# unit_theta = np.array([
#     [math.cos(x[2, 0])],
#     [math.sin(x[2, 0])]
# ])

# # Find vector from robot to desired point
# ref = x_d[:2]-x[:2]

# # Steering error is the cross-product
# e = np.cross(unit_theta.T, ref.T)[0]
# print("ref = " + ref)
# print("e = " + e)

line1, line2 = plt.plot([1, 2, 3], [4, 5, 6], [1, 2, 3], [6, 5, 4])

# Change color of the first line to red
line1.set_color('red')

# Set line style of the second line to dashed
line2.set_linestyle('--')

plt.show()
