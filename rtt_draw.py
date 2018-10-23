# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 16:19:51 2018

@author: y84107158
"""
import os
import csv
import glob
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('path',help='log mir')
args = parser.parse_args()
dir = args.path

try:
    os.makedirs(dir+'/rtt_figure')
except:
    pass

def data_smooth(data,interval):
    smooth_data = []
    for i in range(len(data))[::interval]:
        smooth_data.append(sum(data[i:i+interval])/len(data[i:i+interval]))
    return smooth_data

sender_filelist = glob.glob(r''+dir+'/sender_rtt_env_*_tp_*.log')

# print(sender_filelist)

for file_path in sender_filelist:

    df_sender = pd.read_csv(file_path)
    column = df_sender.columns[0]
    sender_packets = float(column)

    filename = os.path.basename(file_path)
    print(filename)

    df_receiver = pd.read_csv(dir+'/receiver'+filename[10:])
    receiver_packets = float(df_receiver.columns[0])

    loss_rate = (sender_packets-receiver_packets)/sender_packets

    df_sender = df_sender[11:]
    df_sender = df_sender.astype(float)

    sns.set_style("whitegrid")
    plt.plot(data_smooth(df_sender[column].tolist(),int(len(df_sender)/200)),color='royalblue',linewidth=1)
    plt.title('RTT')
    plt.xlabel('Time')
    plt.ylabel('Time(ms)')
    plt.savefig(dir+'/rtt_figure/'+filename+'_rtt.png',dpi=1000)
    plt.close()
    '''
    plt.plot(df_sender[column].tolist(),color='cornflowerblue',linewidth=0.75)
    plt.title('RTT')
    plt.xlabel('Time')
    plt.ylabel('Time(ms)')
    plt.savefig(dir+'/rtt_figure/'+filename+'_rtt_.png',dpi=1000)
    plt.close()
    '''
    #sns.boxplot(df_sender[column].tolist(), showmeans=True, showfliers=False,medianprops = {'linestyle':'--','color':'orange'})
    sns.boxplot(y = df_sender[column], showmeans=True, showfliers=False, medianprops = {'linestyle':'-','color':'red'})
    plt.title('RTT box')
    plt.ylabel('Time(ms)')
    plt.savefig(dir+'/rtt_figure/'+filename+'_rtt_box.png',dpi=1000)
    plt.close()
    

    rtt_mean = np.mean(np.array(df_sender)[1:])
    ptile = np.percentile(df_sender,95)
    d_loss = ["loss rate", loss_rate]
    d_mean = ["mean", rtt_mean]
    d_ptile = ["95 percentile",ptile]
    
    csvFile = open(os.path.join(dir,filename+"_rtt.csv"), "w")
    writer = csv.writer(csvFile)

    writer.writerow(d_loss)
    writer.writerow(d_mean)
    writer.writerow(d_ptile)

    csvFile.close()
