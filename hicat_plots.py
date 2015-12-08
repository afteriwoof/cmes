
'''
Project :       HELCATS
Name    :       hicat_plots
Purpose :       Produce exploratory plots of the HI CME catalogs.
Explanation:    Input the HI CME catalogs as dataframes from the module cmes.py and use matplotlib to produce histograms and scatterplots etc of their parameters such as speeds and position angles.
Use     :       $ python hicat_plots.py
		> import hicat_plots
		> hicat_plots.run_all()
Inputs  :       
Outputs :       png files
Keywords:       
Calls   :       os, numpy, matplotlib, pandas
		config, savefig, cmes
Written :       Jason P Byrne, STFC/RAL Space, Dec 2015 (jason.byrne@stfc.ac.uk)
Revisions:
2015-12-08 JPB : 
'''

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import config

from savefig import save

from cmes import hicat

# Call the catalog functions:
#df_cdaw = cdaw().convert_objects(convert_numeric=True)
#df_hicact_a = hicact('A').convert_objects(convert_numeric=True)
#df_hicact_b = hicact('B').convert_objects(convert_numeric=True)
df_hicat = hicat().convert_objects(convert_numeric=True)

# Drop NaNs
# df.dropna(axis='rows',how='any',inplace=True)
# df_cdaw.describe()
# Generate some initial plots for CDAW
# df_cdaw.hist()
# save(path=os.path.join(config.wp3_path,"cdaw_cme_catalog/cdaw_hist"),verbose=True)
# plt.show()

# global variables
binwidth=50 
colors = ['r','b','grey']
labels = ['Ahead','Behind','']
speeds_xlim = [0,2100]
speeds_xlabel = "Speed ($km s^{-1}$)"

# Histogram of STEREO-Ahead/Behind speeds
def hi_spc_speeds(v,**kwargs):
	if kwargs:
		print kwargs
	if 'spc' in kwargs:
		spc = kwargs['spc']
	else:
		spc = -1
	if 'tit' in kwargs:
		tit = kwargs['tit']
	else:
		tit = ""
	v = np.array(v.astype('float'))
	plt.hist(v,bins=np.arange(0, max(v) + binwidth, binwidth),color=colors[spc])
	plt.title("%s STEREO %s" %(tit.upper(),labels[spc]))
	plt.xlim(speeds_xlim)
	plt.xlabel(speeds_xlabel)
	save(path=os.path.join(config.hicat_path,"%s_speeds_hist_%s" %(tit,labels[spc])),verbose=True)


# Histogram of STEREO-Ahead & Behind speeds
def hicat_speeds(v_a,v_b):
	plt.hist(v_a,bins=np.arange(0,max(v_a)+binwidth,binwidth),histtype='stepfilled',\
		normed=False,color='r',label='Ahead')
	plt.hist(v_b,bins=np.arange(0,max(v_b)+binwidth,binwidth),histtype='stepfilled',\
		normed=False,color='b',alpha=0.5,label='Behind')
	plt.title("HICACTus CME Speeds")
	plt.xlim(speeds_xlim)
	plt.xlabel(speeds_xlabel)
	plt.ylabel("Count")
	plt.legend(prop={'size':8})
	save(path=os.path.join(config.hicat_path,"hicat_speeds_hist"),verbose=True)

# Scatterplot of STEREO-Ahead/Behind speeds against position angles
def hicat_spc_speeds_pa(df_hicat,spc):
	plt.scatter(df_hicat.v,df_hicat.pa,s=80,facecolor=colors[spc],edgecolor='none',alpha=0.25)
	#plt.axhline(y=df_hicat_b.pa.min())
	#plt.axhline(y=df_hicat_b.pa.max())
	plt.xlim(speeds_xlim)
	plt.title("HICACTus STEREO-%s" %labels[spc])
	plt.xlabel(speeds_xlabel)
	plt.ylabel("Position Angle ($deg$)")
	save(path=os.path.join(config.hicat_path,"hicat_speeds_pa_%s" %labels[spc]),verbose=True)

# Scatterplot of STEREO-Ahead & Behind speeds against position angles
def hicat_speeds_pa():
	plt.figure(num=None,figsize=(6,8),dpi=80,facecolor='w',edgecolor='k')
	a = plt.scatter(df_hicat_a.v,df_hicat_a.pa,s=20,facecolor='red',edgecolor='none',alpha=0.25)
	b = plt.scatter(df_hicat_b.v,df_hicat_b.pa,s=20,facecolor='blue',edgecolor='none',alpha=0.25)
	plt.xlim(speeds_xlim)
	plt.ylim([0,360])
	plt.title("HICACTus STEREO CMEs")
	plt.xlabel(speeds_xlabel)
	plt.ylabel("Position Angle ($deg$)")
	plt.legend([a,b],['Ahead','Behind'],prop={'size':8})
	save(path=os.path.join(config.hicat_path,"hicat_speeds_pa"),verbose=True)

'''
wp3_speeds=df_hicat[['FP speed [kms-1]','SSE speed [kms-1]','HM speed [kms-1]']]
wp3_speeds.plot(kind='hist',stacked=True,bins=100)
save(path=os.path.join(config.hicat_path,"wp3_speeds_hist"),verbose=True)
'''

# Split speeds by year
def hicat_speeds_datetime():
	import datetime
	time_format = "%Y/%m/%d %H:%M"
	datetimes_a = np.array([datetime.datetime.strptime(x,time_format) for x in df_hicat_a.starttime])
	datetimes_b = np.array([datetime.datetime.strptime(x,time_format) for x in df_hicat_b.starttime])
	plt.figure(num=None,figsize=(6,8),dpi=80,facecolor='w',edgecolor='k')
	a = plt.scatter(df_hicat_a.v,datetimes_a,s=20,facecolor='red',edgecolor='none',alpha=0.25)
	b = plt.scatter(df_hicat_b.v,datetimes_b,s=20,facecolor='blue',edgecolor='none',alpha=0.25)
	plt.xlim(speeds_xlim)
	plt.title("HICACTus STEREO CMEs")
	plt.xlabel(speeds_xlabel)
	plt.ylabel("Time")
	plt.legend([a,b],['Ahead','Behind'],prop={'size':8})
	save(path=os.path.join(config.hicat_path,"hicat_speeds_datetimes"),verbose=True)

def fourier_speeds():
	from scipy.fftpack import fft
	yf = fft(df_hicat_b.v)
	plt.scatter(df_hicat_b.v,yf)
	#plt.show()

def run_all():
	hi_spc_speeds(df_hicat_a.v,spc=0,tit='hicat')
	hi_spc_speeds(df_hicat_b.v,spc=1,tit='hicat')
	hicat_speeds(df_hicat_a.v, df_hicat_b.v)
	hicat_spc_speeds_pa(df_hicat_a,0)
	hicat_spc_speeds_pa(df_hicat_b,1)
	hicat_speeds_pa()
	hicat_speeds_datetime()

	hi_spc_speeds(df_hicat[["FP speed [kms-1]"]],tit='hicat',ledge=')



