import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal


def triangle_pulse(t, width=2):
    return np.maximum(0, 1-abs(t)/width/2)



#Question 1A

wm = 3*math.pi
t_list = np.arange(0.2, 3, 0.01)
x = lambda t: 4/(wm*math.pi*t**2)*(math.sin(wm*t))**2*math.cos(wm*t)*math.sin(2*wm*t)
x_list = [abs(x(t)) for t in t_list]

plt.plot(t_list, x_list)
plt.xlabel('Time [s]')
plt.ylabel('|x(t)| [V]')
plt.title('Absolute Value of x(t)')
plt.show()


# Question 1B
w_list = np.arange(-17*math.pi, 17*math.pi, 0.1)

xF = lambda w: (4 * wm / math.pi* 1j) * (-1 * triangle_pulse(((w + 3 * wm) / 2 * wm),4 * wm) - \
                                     triangle_pulse(((w + 1 * wm) / 2 * wm), 4 * wm) + \
                                     triangle_pulse(((w - 1 * wm) / 2 * wm), 4 * wm) + \
                                     triangle_pulse(((w - 1 * wm) / 2 * wm), 4 * wm))

xF_list = [np.abs(xF(w)) for w in w_list]


# plt.plot(w_list, xF_list)
# plt.xlabel('Omega [rad/s]')
# plt.ylabel('|xF(w)|')
# plt.title('Absolute Value of xF(w)')
# plt.show()

#Question 1C

w_max = 5*wm
ws = 2*w_max
Ts = 2*math.pi/ws

t_n = np.arange(0.2,3,Ts)
x_n_list = [x(t) for t in t_n]

plt.plot(t_n, x_n_list, 'o')
plt.xlabel('Time [s]')
plt.ylabel('|x(t)| [V]')
plt.title('Absolute Value of x(t)')
plt.show()





