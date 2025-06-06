import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Create a figure
fig = plt.figure(figsize=(10, 10))  # Set the overall figure size

# Create a GridSpec layout (2 rows, 2 columns)
gs = GridSpec(2, 2, figure=fig)

# Create subplots with different sizes using the GridSpec layout
ax1 = fig.add_subplot(gs[:, 0])  # Top-left and Bottom Left Space Subplot
ax2 = fig.add_subplot(gs[0, 1])  # Top-right subplot (regular size)
ax3 = fig.add_subplot(gs[1, 0])  # Bottom-left subplot (regular size)
ax3.axis('off')
ax4 = fig.add_subplot(gs[1, 1])  # Bottom-right subplot (regular size)

# Increase the size of the first subplot (make it span both rows)
# Increase size of the first subplot by spanning both rows


# Plot some data
ax1.plot([0, 1, 2], [0, 1, 4])
ax1.set_title("Large Plot 1")

ax2.plot([0, 1, 2], [0, -1, -2])
ax2.set_title("Plot 2")


ax4.plot([0, 1, 2], [0, 0, 0])
ax4.set_title("Plot 4")

# Set the aspect ratio of the first subplot (large subplot) to equal
ax1.set_aspect('equal')

# Adjust layout to avoid overlap
plt.tight_layout()

# Show the plot
plt.show()
