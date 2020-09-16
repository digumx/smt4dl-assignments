import numpy as np
import matplotlib.pyplot as plt
import random
import math

n1 = 10
n2 = 20
amp = 4
time = [i/1000.0 for i in range (0,1000)] # for one second
# let angular velocity be 1 radian per second
wave1 = [amp*math.sin(n1*t)  for t in time]
wave2 = [amp*math.sin(n2*t)  for t in time]
#superposition of two waves is obtained by adding their values at each instance of time
superposition = [wave1[i] + wave2[i] for i in range(len(time))]
plt.plot(time, wave1, 'b--', label = "Wave #1")
plt.plot(time,wave2, 'g',  label = "Wave #2")
plt.plot(time, superposition, 'r-',  label = "Superposition of wave #1 and wave #2")
plt.legend()
plt.xlabel("Time - t")
plt.ylabel("Dispacement - x")
plt.title("Superposition of two waves")
plt.grid()
plt.show()


