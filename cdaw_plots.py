
'''
Project :       HELCATS
Name    :       cdaw_plots
Purpose :       Produce exploratory plots of the CDAW CME catalog.
Explanation:    Input the CDAW CME catalog as a dataframe from the module cmes.py and use matplotlib to produce histograms and scatterplots etc of their parameters such as speeds and position angles.
Use     :       $ python cdaw_plots.py
		> import cdaw_plots
		> cdaw_plots.run_all()
Inputs  :       
Outputs :       png files
Keywords:       
Calls   :       os, numpy, matplotlib, pandas
		config, savefig, cmes
Written :       Jason P Byrne, STFC/RAL Space, Dec 2015 (jason.byrne@stfc.ac.uk)
Revisions:
2015-12-10 JPB : 
'''

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import config

from savefig import save

from cmes import cdaw
# Call the catalog functions:
df_cdaw = cdaw().convert_objects(convert_numeric=True)

# Drop NaNs
# df.dropna(axis='rows',how='any',inplace=True)
# df_cdaw.describe()
# Generate some initial plots for CDAW
# df_cdaw.hist()
# save(path=os.path.join(config.wp3_path,"cdaw_cme_catalog/cdaw_hist"),verbose=True)
# plt.show()

# global variables
binwidth=50 
speeds_lim = [0,4000]
speeds_label = "Speed ($km s^{-1}$)"
ledge_sz = 10

cols_hist = ['cpa','mpa','width','lin_speed','quad_speed_init','quad_speed_20',\
	'quad_speed_final','accel']#'kin_energy','mass'
labels_hist = ['Central PA','Measurement PA','Ang. Width','Linear Speed','Quad. Speed (initial)']
cols_boxplot = ['lin_speed','quad_speed_init','quad_speed_final','quad_speed_20']
labels_boxplot = ['Linear Speed','Quad. Speed (initial)','Quad. Speed (final)','Quad. Speed (20 R_Sun)']
# Histograms of all CDAW parameters
def cdaw_hists():
	fig = plt.figure(1,figsize=(20,20))
	df_cdaw.hist(column=cols_hist,bins=50,grid=False)
	save(path=os.path.join(config.cdaw_path,"cdaw_hist"),verbose=True)
	#boxplot
	fig = plt.figure(2,figsize=(9,6))
	ax = fig.add_subplot(111)
	df_cdaw.boxplot(column=cols_boxplot,grid=False)
	ax.set_xticklabels(labels_boxplot)
	plt.ylabel(speeds_label)
	plt.title("CDAW LASCO CME Catalog")
	plt.tight_layout()
	save(path=os.path.join(config.cdaw_path,"cdaw_speeds_boxplot"),verbose=True)

# Histograms of CDAW CME position angles
def cdaw_pa():
	binwidth=5
	plt.figure(num=None,figsize=(8,10),dpi=80,facecolor='w',edgecolor='k')
	plt.subplot(211)
	plt.hist(df_cdaw.cpa.values,bins=np.arange(0,max(df_cdaw.cpa)+binwidth,binwidth),normed=False)
	plt.xlim([0,360])
	plt.xlabel("Position Angle ($deg.$)")
	plt.ylabel("Count")
	plt.title("CDAW CMEs Central Position Angle")
	plt.subplot(212)
	plt.hist(df_cdaw.mpa.values,bins=np.arange(0,max(df_cdaw.mpa)+binwidth,binwidth),normed=False)
	plt.xlim([0,360])
	plt.xlabel("Position Angle ($deg.$)")
	plt.ylabel("Count")
	plt.title("CDAW CMEs Measured Position Angle")
	plt.tight_layout()
	save(path=os.path.join(config.cdaw_path,"cdaw_pa"),verbose=True)

# Scatterplot of CDAW speeds against position angles
def cdaw_speeds_pa():
	plt.figure(figsize=(10,8),dpi=80,facecolor='w')
	ssz = 10
	#lin = plt.scatter(df_cdaw.lin_speed,df_cdaw.mpa,s=ssz,facecolor='red',\
	#	edgecolor='none',alpha=0.3)
	#q_i = plt.scatter(df_cdaw.quad_speed_init,df_cdaw.mpa,s=ssz,facecolor='red',\
	#	edgecolor='none',alpha=0.3)
	q_f = plt.scatter(df_cdaw.quad_speed_final,df_cdaw.mpa,s=ssz,facecolor='blue',\
		edgecolor='none',alpha=0.3)
	plt.xlim([0,3500])
	plt.ylim([0,360])
	plt.title("CDAW LASCO CME Catalog")
	plt.xlabel(speeds_label)
	plt.ylabel("Measured Position Angle ($deg.$)")
	#plt.legend([lin,q_i,q_f],['Linear','Quad. (init.)','Quad. (final)'],prop={'size':ledge_sz})
	#plt.legend([lin,q_f],['Linear','Quad. (final)'],prop={'size':ledge_sz})
	plt.legend([q_f],['Quad. (final)'],prop={'size':ledge_sz})
	save(path=os.path.join(config.cdaw_path,"cdaw_speeds_pa"),verbose=True)


# Split speeds by year
def cdaw_speeds_datetime():
	import datetime
	import matplotlib.ticker as ticker
	time_format = "%Y/%m/%dT%H:%M:%S"
	datetimes = np.array([datetime.datetime.strptime(x,time_format) \
		for x in df_cdaw.date.values+'T'+df_cdaw.time.values])
	plt.figure(num=None,figsize=(10,7),dpi=80,facecolor='w',edgecolor='k')
	ssz=10
	#lin = plt.scatter(datetimes,df_cdaw.lin_speed,s=ssz,facecolor='red',\
	#	edgecolor='none',alpha=0.3)
	q_f = plt.scatter(datetimes,df_cdaw.quad_speed_final,s=ssz,facecolor='blue',\
		edgecolor='none',alpha=0.3)
	plt.ylim([0,3500])
	plt.title("CDAW LASCO CME Catalog")
	plt.ylabel(speeds_label)
	plt.xlabel("Time")
	ax = plt.axes()
	ax.xaxis.set_major_locator(ticker.MultipleLocator(365))
	labels=ax.get_xticklabels()
	plt.setp(labels,rotation=40)
	plt.legend([q_f],['Quad. (final)'],prop={'size':ledge_sz})
	#plt.legend([lin,q_f],['Linear','Quad. (final)'],prop={'size':ledge_sz})
	save(path=os.path.join(config.cdaw_path,"cdaw_speeds_datetimes"),verbose=True)

cdaw_speeds_datetime()

# Run All
def run_all():
	cdaw_hists()
	cdaw_pa()
	cdaw_speeds_pa()

