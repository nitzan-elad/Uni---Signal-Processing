import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal


def triangle_pulse(t, width=2):
    return np.maximum(0, 1-abs(t)/width/2)

def create_sampled(x,t,Ts):
    t_sampled = np.arange(t[0], t[-1], Ts)
    x_sampled = np.interp(t_sampled, t, x)
    return t_sampled, x_sampled

def create_zoh_array(x, t, Ts):
    x_zoh = np.zeros_like(t)
    t_smp, x_smp = create_sampled(x, t, Ts)
    for i in range(len(t_smp)-1):
        x_zoh[(t>=t_smp[i]) & (t < t_smp[i+1])] = x_smp[i]
    x_zoh[t>=t_smp[-1]] = x_smp[-1]
    return t, x_zoh


#Question 1A

wm = 3*math.pi
t_start = 0.2
t_end = 3
t_step = 0.01
t_list = np.arange(t_start, t_end, t_step)
x = lambda t: 4/(wm*math.pi*t**2)*(math.sin(wm*t))**2*math.cos(wm*t)*math.sin(2*wm*t)
x_list = [(x(t)) for t in t_list]
abs_x_list = [abs(x(t)) for t in t_list]
# plt.legend()
# plt.plot(t_list, abs_x_list, label='|x(t)|')
# plt.xlabel('Time [s]')
# plt.ylabel('|x(t)| [V]')
# plt.title('1A - Absolute Value of x(t)')
# plt.grid(True)
# plt.show()


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
# plt.title('1B - Absolute Value of xF(w)')
#plt.grid(True)
# plt.show()

#Question 1C

w_max = 5*wm
ws = 2*w_max
Ts = 1/15

# t_ZOH, x_ZOH = create_zoh_array(x_list, t_list, Ts)
# plt.plot(t_ZOH, x_ZOH, color='orange', label='ZOH of x(t)')
# plt.plot(t_list, x_list, label='x(t)')
# plt.legend()
# plt.xlabel('Time [s]')
# plt.ylabel('x(t) [V]')
# plt.title('1C - Zero Order Hold (ZOH)')
# plt.grid(True)
# plt.show()


xF_ZOH = lambda w,k: xF(w-k*2*np.pi/Ts)

xF_ZOH_list = [0 for w in range(len(w_list))]

for i in range(len(w_list)):
    for k in [-2,-1]:
        xF_ZOH_list[i] += xF_ZOH(w_list[i], k)
    xF_ZOH_list[i] = (1/1j)*np.exp(-1j*w_list[i]*Ts/2)*np.sinc(w_list[i]*Ts/(2*np.pi))*xF_ZOH_list[i]


abs_xF_ZOH_list = [abs(xF_ZOH_list[i]) for i in range(len(w_list))]
plt.plot(w_list, abs_xF_ZOH_list)
plt.show()








