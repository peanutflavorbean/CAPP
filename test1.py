import csv
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
file = open('./Rrod_tolerance.csv','r')
rdr = csv.reader(file)
Rod39 = [[],[],[],[]]
Rod40 = [[],[],[],[]]
Rod41 = [[],[],[],[]]

for line in rdr:
    if (int(line[1]) == 41):
        Rod41[0].append(float(line[2]))
        Rod41[1].append(float(line[4]))
        Rod41[2].append(float(line[-2]))
        Rod41[3].append(float(line[-1]))
    elif (int(line[1]) == 39):
        Rod39[0].append(float(line[2]))
        Rod39[1].append(float(line[4]))
        Rod39[2].append(float(line[-2]))
        Rod39[3].append(float(line[-1]))
    elif (int(line[1]) == 40):
        Rod40[0].append(float(line[2]))
        Rod40[1].append(float(line[4]))
        Rod40[2].append(float(line[-2]))
        Rod40[3].append(float(line[-1]))
file.close()

for i in range(0,4):
    Rod39[i].reverse()
    Rod40[i].reverse()
    Rod41[i].reverse()

def func_3(x,a,b,c,d):
    return a*np.power(x,3)+b*np.power(x,2)+c*np.power(x,1)+d
def func_2(x,a,b,c):
    return a*np.power(x,2)+b*np.power(x,1)+c

x_39, x_40, x_41 = Rod39[0], Rod40[0], Rod41[0]
y1_39, y1_40, y1_41 = Rod39[1], Rod40[1], Rod41[1]
freq_range = np.linspace(1180,1500,320)
plt.plot(x_39,y1_39,'ro')
plt.plot(x_40,y1_40,'bo')
plt.plot(x_41,y1_41,'go')
popt_39, pcov_39 = curve_fit(func_3,x_39,y1_39)
plt.plot(freq_range,func_3(freq_range,*popt_39),'r-')
popt_40, pcov_40 = curve_fit(func_3,x_40,y1_40)
plt.plot(freq_range,func_3(freq_range,*popt_40),'b-')
popt_41, pcov_41 = curve_fit(func_3,x_41,y1_41)
plt.plot(freq_range,func_3(freq_range,*popt_41),'g-')
print(np.diag(pcov_39))
plt.show()
