
'''
Project :       HELCATS
Name    :       hicact_plots
Purpose :       Produce exploratory plots of the CME CACTus catalogs.
Explanation:    Input the CACTUS CME catalogs as dataframes from the module cmes.py and use matplotlib to produce histograms and scatterplots etc of their parameters such as speeds and position angles.
Use     :       $ python hicact_plots.py
		> import hicact_plots
		> hicact_plots.run_all()
Inputs  :       
Outputs :       png files
Keywords:       
Calls   :       os, numpy, matplotlib, pandas
		config, savefig, cmes
Written :       Jason P Byrne, STFC/RAL Space, Dec 2015 (jason.byrne@stfc.ac.uk)
Revisions:
2015-12-09 JPB : Updates to the code for better plots.
'''

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import config

from savefig import save

from cmes import cdaw,hicact,hicat
# Call the catalog functions:
df_hicact_a = hicact('A').convert_objects(convert_numeric=True)
df_hicact_b = hicact('B').convert_objects(convert_numeric=True)

# global variables
binwidth=50 
colors = ['r','b','grey']
labels = ['Ahead','Behind','']
speeds_lim = [0,2100]
speeds_label = "Speed ($km s^{-1}$)"
ledge_sz = 10
alph = 0.35

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
	plt.xlim(speeds_lim)
	plt.xlabel(speeds_label)
	save(path=os.path.join(config.hicact_path,"%s_speeds_hist_%s" %(tit,labels[spc])),verbose=True)


# Histogram of STEREO-Ahead & Behind speeds
def hicact_speeds(v_a,v_b):
	plt.hist(v_a,bins=np.arange(0,max(v_a)+binwidth,binwidth),histtype='stepfilled',\
		normed=False,color='r',label='Ahead')
	plt.hist(v_b,bins=np.arange(0,max(v_b)+binwidth,binwidth),histtype='stepfilled',\
		normed=False,color='b',alpha=alph,label='Behind')
	plt.title("HICACTus CME Speeds")
	plt.xlim(speeds_lim)
	plt.xlabel(speeds_label)
	plt.ylabel("Count")
	plt.legend(prop={'size':8})
	save(path=os.path.join(config.hicact_path,"hicact_speeds_hist"),verbose=True)


# Scatterplot of STEREO-Ahead/Behind speeds against position angles
def hicact_spc_speeds_pa(df_hicact,spc):
	plt.scatter(df_hicact.v,df_hicact.pa,s=80,facecolor=colors[spc],edgecolor='none',alpha=alph)
	#plt.axhline(y=df_hicact_b.pa.min())
	#plt.axhline(y=df_hicact_b.pa.max())
	plt.xlim(speeds_lim)
	plt.title("HICACTus STEREO-%s" %labels[spc])
	plt.xlabel(speeds_label)
	plt.ylabel("Position Angle ($deg$)")
	save(path=os.path.join(config.hicact_path,"hicact_speeds_pa_%s" %labels[spc]),verbose=True)


# Scatterplot of STEREO-Ahead & Behind speeds against position angles
def hicact_speeds_pa():
	plt.figure(num=None,figsize=(6,8),dpi=80,facecolor='w',edgecolor='k')
	a = plt.scatter(df_hicact_a.v,df_hicact_a.pa,s=20,facecolor='red',edgecolor='none',alpha=alph)
	b = plt.scatter(df_hicact_b.v,df_hicact_b.pa,s=20,facecolor='blue',edgecolor='none',alpha=alph)
	plt.xlim(speeds_lim)
	plt.ylim([0,360])
	plt.title("HICACTus STEREO CMEs")
	plt.xlabel(speeds_label)
	plt.ylabel("Position Angle ($deg$)")
	plt.legend([a,b],['Ahead','Behind'],prop={'size':8})
	save(path=os.path.join(config.hicact_path,"hicact_speeds_pa"),verbose=True)


# Split speeds by year
def hicact_speeds_datetime(df_hicact_a,df_hicact_b):
	import datetime
	time_format = "%Y/%m/%d %H:%M"
	datetimes_a = np.array([datetime.datetime.strptime(x,time_format) for x in df_hicact_a.starttime])
	datetimes_b = np.array([datetime.datetime.strptime(x,time_format) for x in df_hicact_b.starttime])
	plt.figure(num=None,figsize=(7,8),dpi=80,facecolor='w',edgecolor='k')
	a = plt.scatter(datetimes_a,df_hicact_a.v,s=20,facecolor='red',edgecolor='none',alpha=alph)
	b = plt.scatter(datetimes_b,df_hicact_b.v,s=20,facecolor='blue',edgecolor='none',alpha=alph)
	plt.ylim([0,2300])
	plt.title("HICACTus CMEs")
	plt.ylabel(speeds_label)
	plt.xlabel("Time")
	plt.legend([a,b],['Ahead','Behind'],prop={'size':ledge_sz},loc=2)
	save(path=os.path.join(config.hicact_path,"hicact_speeds_datetimes"),verbose=True)

# Run all
def run_all():
	hi_spc_speeds(df_hicact_a.v,spc=0,tit='hicact')
	hi_spc_speeds(df_hicact_b.v,spc=1,tit='hicact')
	hicact_speeds(df_hicact_a.v, df_hicact_b.v)
	hicact_spc_speeds_pa(df_hicact_a,0)
	hicact_spc_speeds_pa(df_hicact_b,1)
	hicact_speeds_pa()
	hicact_speeds_datetime(df_hicact_a,df_hicact_b)



