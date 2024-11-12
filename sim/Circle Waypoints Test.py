import numpy as np
import matplotlib.pyplot as plt

# Circle parameters
radius = 5
center_x, center_y = 0, 0  # Circle center coordinates
num_waypoints = 50         # Number of waypoints

# Generate angles for waypoints evenly spaced around the circle
# array of all angles values around circle
angles = np.linspace(0, 2 * np.pi, num_waypoints, endpoint=False)


# Calculate x and y coordinates of waypoints
x_points = center_x + radius * np.cos(angles)  # array of x coordinate values
y_points = center_y + radius * np.sin(angles)  # array of y coordinate values

print(x_points, ";", y_points, ";")

# Plotting the circle and waypoints
plt.figure(figsize=(6, 6))
# Blue circle with lines
plt.plot(x_points, y_points, 'bo-', label="Waypoints")
plt.scatter([center_x], [center_y], color='red',
            label="Center")  # Red center point
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Circle of Waypoints")
plt.legend()
plt.axis("equal")
plt.grid(True)
plt.show()
