import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import triang


#Task 1 - Sampling and Reconstruction
wm = 3*math.pi
t_list = np.arange(0.2, 3, 0.01)
x = lambda t: 4/(wm*math.pi*t**2)*(math.sin(wm*t))**2*math.cos(wm*t)*math.sin(2*wm*t)
x_list = [abs(x(t)) for t in t_list]

plt.plot(t_list, x_list)
plt.xlabel('Time [s]')
plt.ylabel('|x(t)| [V]')
plt.title('Absolute Value of x(t)')
plt.show()


