# -*- coding: utf-8 -*-

import csv
from sklearn.metrics import r2_score
import numpy as np
import scipy.stats as stats
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument('path',help='log mir')
args = parser.parse_args()
dir = args.path

try:
	os.makedirs(dir+'/log_figure')
except:
	pass

def data_smooth(data,interval):
	smooth_data = []
	for i in range(len(data))[::interval]:
		smooth_data.append(sum(data[i:i+interval])/len(data[i:i+interval]))
	return smooth_data

sns.set_style("whitegrid")

for path,dirlist,filelist in os.walk(dir):
	for file in filelist:
		if os.path.splitext(file)[1] == '.log':
			filename = file
			file_path = os.path.join(path,file)
			df = pd.read_csv(file_path)
			print('draw',file_path)
			sd_best_cwnd = data_smooth(df[' best_cwnd'],10)
			sd_indigo_cwnd = data_smooth(df[' indigo_cwnd'],10)
			
			#Cosine
			cosi=np.dot(np.array(sd_best_cwnd),\
						np.array(sd_indigo_cwnd))/(np.linalg.norm(np.array(\
								sd_best_cwnd))*(np.linalg.norm(np.array(sd_indigo_cwnd))))

			#mae
			mae=np.mean(np.abs(np.array(sd_best_cwnd)-np.array(sd_indigo_cwnd)))
			#mse
			df_mse=np.array(sd_best_cwnd)-np.array(sd_indigo_cwnd)
			mse=np.mean(np.square(df_mse))
			#r_square
			r_square=r2_score(sd_best_cwnd, sd_indigo_cwnd)  
			#pearson coefficent
			pearsonr=stats.pearsonr(sd_best_cwnd,sd_indigo_cwnd)[0]
			d4 = ["Cosine",cosi]
			d5 = ["Mae", mae]
			d6 = ["Mse", mse]
			d8 = ["R_square",r_square]
			d9 = ["Pearson Coefficent",pearsonr]

			csvFile = open(os.path.join(dir,filename+"_indigo_cwnd_best_cwnd_fitting result.csv"), "w")
			writer = csv.writer(csvFile)
			writer.writerow(d4)
			writer.writerow(d5)
			writer.writerow(d6)
			writer.writerow(d8)
			writer.writerow(d9)
			csvFile.close()
			
			
			plt.plot(sd_best_cwnd,color='b',linewidth=1,label='Best cwnd')
			plt.plot(sd_indigo_cwnd,color='firebrick',linewidth=1,label='Indigo cwnd')
			plt.legend()
			plt.title('Best cwnd vs Indigo cwnd')
			plt.xlabel('Time')
			plt.ylabel('Window size(Byte)')
			plt.savefig(path+'/log_figure/'+file+'_best_vs_indigo.png',dpi=1000)
			plt.close()


			df['total'] = df.ix[:,[' tg_traffic',' sender_traffic']].apply(lambda x:x.sum(), axis =1)
			sd_tg_traffic = data_smooth(df[' tg_traffic'],10)
			sd_sender_traffic = data_smooth(df[' sender_traffic'],10)
			sd_total = data_smooth(df['total'],10)
			plt.plot(sd_tg_traffic,linestyle='--',color='b',linewidth=1,label='Background traffic')
			plt.plot(sd_sender_traffic,linestyle='--',color='firebrick',linewidth=1,label='Indigo traffic')
			plt.plot(sd_total,color='green',linewidth=1,label='Total traffic')
			plt.legend()
			plt.title('Background traffic vs Indigo traffic')
			plt.xlabel('Time')
			plt.ylabel('TX Rate(Mbps)')
			plt.savefig(path+'/log_figure/'+file+'_tg_vs_sender.png',dpi=1000)
			plt.close()


			df_tg_traffic = df[' tg_traffic']
			sd_queueing_factor = data_smooth(df[' queueing_factor'],10)
			
			sd_queueing_factor_array=np.array(sd_queueing_factor)
			queueing_factor_pdf=sns.distplot(sd_queueing_factor_array)
			queueing_factor_pdf.set_title("PDF of Queueing factor")
			pdf_figure = queueing_factor_pdf.get_figure()    
			pdf_figure.savefig(path+'/log_figure/'+file+'PDF_queueing_factor.png', dpi=1000)
			plt.close()
			
			queueing_factor_cdf=sns.distplot(sd_queueing_factor_array,hist_kws=dict(cumulative=True),kde_kws=dict(cumulative=True))
			queueing_factor_cdf.set_title("CDF of Queueing factor")
			cdf_figure = queueing_factor_cdf.get_figure()    
			cdf_figure.savefig(path+'/log_figure/'+file+'CDF_queueing_factor.png', dpi=1000)
			plt.close()
			
			plt.plot(sd_queueing_factor,color='b',linewidth=1,label='tg traffic')
			plt.title('Queueing factor')
			plt.xlabel('Time')
			plt.ylabel('value')
			plt.savefig(path+'/log_figure/'+file+'_queueing_factor.png',dpi=1000)
			plt.close()







