import matplotlib.pyplot as plt
import configparser
import numpy as np
import time

def save_threshold_reward(a,b):
    plt.grid(True, linestyle = "--")#, color = "r", linewidth = "3")
    plt.scatter([x+1 for x in range(len(a))],a,linestyle='-',color = "green")
    plt.scatter([x+1 for x in range(len(b))],b,linestyle='-',color = "red")
    plt.legend()
    plt.show()
    # plt.savefig("data/threshold %d %s r.jpg" % (env.link_capacity_threshold,datetime.datetime.now().strftime('%Y-%m-%d%H%M%S')))
    plt.close('all')

a = [1,7,0,3,0,15,3,0,0,0,0,23,0,0,3,0,15,1,0,6,20]
b = [1,7,0,3,0,15,3,0,0,0,17,23,0,0,3,0,15,1,0,6,0]
c = [x+1 for x in range(len(a))]
#b = [1,7,9,3,5,15,3,2,1,9,17,23,10,2,3,4,15,1,5,6,0]

def convolution(array):
    new_array = []
    l = len(array)
    array = [0,0,0,0]+array+[0,0,0,0]
    for i in range(l):
        target = array[i]*0.2+array[i+1]*0.4+array[i+2]*0.6+array[i+3]*0.8+array[i+4]*2+array[i+5]*0.8+array[i+6]*0.6+array[i+7]*0.4+array[i+8]*0.2
        new_array.append(target)
    return new_array

def dec2bin(num):
    res = []
    while (num!=0):
        res.append(num%2)
        num = int(num/2)
    return [0 for x in range(6-len(res))] + res[::-1]

'''
start = time.time()
time.sleep(1)
print(convolution(a))
print(convolution(b))
end = time.time()
print(end-start)

start = time.time()
print(np.convolve(a,[0.2,0.4,0.6,0.8,2,0.8,0.6,0.4,0.2],mode='valid'))
print(np.convolve(b,[0.2,0.4,0.6,0.8,2,0.8,0.6,0.4,0.2],mode='valid'))
end = time.time()
print(end-start)
save_threshold_reward(a,b)
save_threshold_reward(convolution(a),convolution(b))
save_threshold_reward(np.convolve(a,[0.2,0.4,0.6,0.8,2,0.8,0.6,0.4,0.2],mode='valid'),np.convolve(b,[0.2,0.4,0.6,0.8,2,0.8,0.6,0.4,0.2],mode='valid'))

start = time.time()
dd = 8
print(dec2bin(dd))
end = time.time()
print(end-start)
'''

'''
    cf = configparser.ConfigParser()
    cf.read('config.ini', encoding='utf8')
    res = cf.getfloat('dqn', 'min_replay_history')
    print(res)
'''



import numpy as np
import sys
import time
import random


v = np.random.uniform(32, )
print(v)