import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal


def triangle_pulse(t, width=2):
    return np.maximum(0, 1-abs(t)/width/2)

def create_zoh_array(x, t, Ts, start=0):
    zoh = [x[0]]
    for i in range(len(t)-1):
        if start + t[i] % Ts == 0:
            zoh.append(x[i])
        else:
            zoh.append(zoh[i-1])
    return zoh


#Question 1A

wm = 3*math.pi
t_start = 0.2
t_end = 3
t_step = 0.01
t_list = np.arange(t_start, t_end, t_step)
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
Ts = 1/15

x_ZOH = create_zoh_array(x_list,t_list, Ts, t_start)
plt.plot(t_list, x_ZOH, 'o')
plt.xlabel('Time [s]')
plt.ylabel('|x(t)| [V]')
plt.title('Absolute Value of x(t)')
plt.show()





