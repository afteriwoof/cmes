
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
	#lin = plt.scatter(df_cdaw.lin_speed,df_cdaw.mpa,s=10,facecolor='red',\
	#	edgecolor='none',alpha=0.3)
	#q_i = plt.scatter(df_cdaw.quad_speed_init,df_cdaw.mpa,s=10,facecolor='red',\
	#	edgecolor='none',alpha=0.3)
	q_f = plt.scatter(df_cdaw.quad_speed_final,df_cdaw.mpa,s=10,facecolor='blue',\
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

cdaw_speeds_pa()

# Histogram of CDAW linear speeds
def hi_geom_speeds(v,**kwargs):
	if kwargs:
		print kwargs
	v = np.array(v.astype('float'))
	plt.hist(v,bins=np.arange(0, max(v) + binwidth, binwidth),color=colors[spc])
	plt.title("%s STEREO %s" %(tit,labels[spc]))
	#plt.xlim(speeds_lim)
	plt.xlabel(speeds_label)
	save(path=os.path.join(config.hicat_path,"%s_speeds_hist_%s" %(tit,labels[spc])),verbose=True)

# Histogram of STEREO-Ahead & Behind speeds
def hicat_all_speeds(*args,**kwargs):
	fp_a = args[0]
	fp_b = args[1]
	sse_a = args[2]
	sse_b = args[3]
	hm_a = args[4]
	hm_b = args[5]
	if 'tit' in kwargs:
		tit = kwargs['tit']
	else:
		tit = ""
	plt.figure(num=None,figsize=(10,12),dpi=80,facecolor='w',edgecolor='k')
	plt.subplot(311)
	plt.hist([fp_a,fp_b],bins=np.arange(0,max(fp_a)+binwidth,binwidth),\
		stacked=True,normed=False,color=colors[0:2],label=labels[0:2])
	plt.title("HICAT CME Speeds\n %s" %tit[0])
	plt.xlim(speeds_lim)
	plt.ylabel("Count")
	plt.legend(prop={'size':ledge_sz})
	plt.subplot(312)
	plt.hist([sse_a,sse_b],bins=np.arange(0,max(sse_a)+binwidth,binwidth),\
                stacked=True,normed=False,color=colors[0:2],label=labels[0:2])
	plt.title("%s" %tit[1])
        plt.xlim(speeds_lim)
        plt.ylabel("Count")
	plt.subplot(313)
        plt.hist([hm_a,hm_b],bins=np.arange(0,max(hm_a)+binwidth,binwidth),\
                stacked=True,normed=False,color=colors[0:2],label=labels[0:2])
	plt.title("%s" %tit[2])
        plt.xlim(speeds_lim)
        plt.xlabel(speeds_label)
        plt.ylabel("Count")
        save(path=os.path.join(config.hicat_path,"hicat_speeds_hist"),verbose=True)


# Scatterplot of STEREO-Ahead/Behind speeds against position angles
def hicat_spc_speeds_pa(df_hicat,spc):
	hm = plt.scatter(df_hicat[['HM speed [kms-1]']],df_hicat[['PA-fit']],s=50,facecolor='green',\
		edgecolor='none',alpha=0.5)
	sse = plt.scatter(df_hicat[['SSE speed [kms-1]']],df_hicat[['PA-fit']],s=50,facecolor='blue',\
		edgecolor='none',alpha=0.5)
	fp = plt.scatter(df_hicat[['FP speed [kms-1]']],df_hicat[['PA-fit']],s=50,facecolor='red',\
		edgecolor='none',alpha=0.5)
	#plt.axhline(y=df_hicat_b.pa.min())
	#plt.axhline(y=df_hicat_b.pa.max())
	plt.xlim(speeds_lim)
	plt.title("HICAT STEREO-%s" %labels[spc])
	plt.xlabel(speeds_label)
	plt.ylabel("Position Angle ($deg$)")
	plt.legend([fp,sse,hm],['Fixed-Phi','Self-Similar Exp.','Harmonic Mean'],prop={'size':ledge_sz})
	save(path=os.path.join(config.hicat_path,"hicat_speeds_pa_%s" %labels[spc]),verbose=True)

# Scatterplot of STEREO-Ahead & Behind speeds against position angles
def hicat_speeds_pa(df_hicat):
	plt.figure(num=None,figsize=(8,10),dpi=80,facecolor='w',edgecolor='k')
        hm = plt.scatter(df_hicat[['HM speed [kms-1]']],df_hicat[['PA-fit']],s=50,facecolor='green',\
                edgecolor='none',alpha=0.35)
        sse = plt.scatter(df_hicat[['SSE speed [kms-1]']],df_hicat[['PA-fit']],s=50,facecolor='blue',\
                edgecolor='none',alpha=0.35)
	fp = plt.scatter(df_hicat[['FP speed [kms-1]']],df_hicat[['PA-fit']],s=50,facecolor='red',\
                edgecolor='none',alpha=0.35)
	plt.xlim(speeds_lim)
	plt.ylim([0,360])
	plt.title("HICAT CMEs")
	plt.xlabel(speeds_label)
	plt.ylabel("Position Angle ($deg$)")
	plt.legend([fp,sse,hm],['Fixed-Phi','Self-Similar Exp.','Harmonic Mean'],prop={'size':ledge_sz})
	save(path=os.path.join(config.hicat_path,"hicat_speeds_pa"),verbose=True)

def hicat_stacked_speeds():
	wp3_speeds=df_hicat[['FP speed [kms-1]','SSE speed [kms-1]','HM speed [kms-1]']]
	wp3_speeds.plot(kind='hist',stacked=True,bins=100)
	save(path=os.path.join(config.hicat_path,"hicat_speeds_stacked"),verbose=True)


# Split speeds by year
def hicat_speeds_datetime(df_hicat):
	import datetime
	time_format = "%Y-%m-%dT%H:%MZ"
	datetimes = np.array([datetime.datetime.strptime(x[0],time_format) \
		for x in df_hicat[['Date [UTC]']].values])
	plt.figure(num=None,figsize=(8,6),dpi=80,facecolor='w',edgecolor='k')
	ssz=30
	hm = plt.scatter(datetimes,df_hicat[['HM speed [kms-1]']],s=ssz,facecolor='green',\
		edgecolor='none',alpha=0.35)
	sse = plt.scatter(datetimes,df_hicat[['SSE speed [kms-1]']],s=ssz,facecolor='blue',\
		edgecolor='none',alpha=0.35)
	fp = plt.scatter(datetimes,df_hicat[['FP speed [kms-1]']],s=ssz,facecolor='red',\
		edgecolor='none',alpha=0.35)
	plt.ylim(speeds_lim)
	plt.title("HICAT CMEs")
	plt.ylabel(speeds_label)
	plt.xlabel("Time")
	plt.legend([fp,sse,hm],['FP','SSE','HM'],prop={'size':ledge_sz},loc=2)
	save(path=os.path.join(config.hicat_path,"hicat_speeds_datetimes"),verbose=True)

# Run All
def run_all():
	cdaw_hists()
	cdaw_pa()


