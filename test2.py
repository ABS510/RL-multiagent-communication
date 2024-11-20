import matplotlib.pyplot as plt
import numpy as np
import time

# Initialize plot
plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots()
x, y = [], []
line, = ax.plot(x, y)

# Update plot in a loop
for i in range(100):
    x.append(i)
    y.append(np.sin(i / 10))
    line.set_xdata(x)
    line.set_ydata(y)
    ax.relim()        # Recalculate limits
    ax.autoscale_view()  # Rescale the view
    plt.draw()
    plt.pause(0.1)  # Pause for 100ms

plt.ioff()  # Turn off interactive mode
plt.show()
