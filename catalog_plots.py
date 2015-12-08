import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import config

from savefig import save

from cmes import cdaw,hicact,hicat

# Call the catalog functions:
df_cdaw = cdaw()
df_hicact_a = hicact('A').convert_objects(convert_numeric=True)
df_hicact_b = hicact('B').convert_objects(convert_numeric=True)
df_hicat = hicat()

# Convert strings to numerics
# df = df.convert_objects(convert_numeric=True)
# Drop NaNs
# df.dropna(axis='rows',how='any',inplace=True)
df_cdaw.describe()
# Generate some initial plots for CDAW
#df_cdaw.hist()
#save(path=os.path.join(config.wp3_path,"cdaw_cme_catalog/cdaw_hist"),verbose=True)
#plt.show()

binwidth=50 #local variable

# Histogram of STEREO-Ahead speeds
def hicact_a_speeds():
	binwidth = 50
	v_a = np.array(df_hicact_a[['v']].astype('float'))
	plt.hist(v_a,bins=np.arange(0, max(v_a) + binwidth, binwidth))
	plt.title("HICACTus STEREO-Ahead")
	save(path=os.path.join(config.hicact_path,"hicact_a_speeds_hist"),verbose=True)

hicact_a_speeds()

# Histogram of STEREO-Behind speeds
def hicact_b_speeds():
	binwidth=1
	v_b = np.array(df_hicact_b[['v']].astype('float'))
	plt.hist(v_b,bins=np.arange(0,max(v_b)+binwidth,binwidth))
	plt.title("HICACTus STEREO-Behind")
	save(path=os.path.join(config.hicact_path,"hicact_b_speeds_hist"),verbose=True)

hicact_b_speeds()

# Histogram of STEREO-Ahead & Behind speeds
def hicact_speeds(v_a,v_b):
	plt.hist(v_a,bins=np.arange(0,max(v_a)+binwidth,binwidth),histtype='stepfilled',\
		normed=False,color='r',label='Ahead')
	plt.hist(v_b,bins=np.arange(0,max(v_b)+binwidth,binwidth),histtype='stepfilled',\
		normed=False,color='b',alpha=0.5,label='Behind')
	plt.title("HICACTus CME Speeds")
	plt.xlabel("Speed [kms-1]")
	plt.ylabel("Count")
	plt.legend(prop={'size':8})
	save(path=os.path.join(config.hicact_path,"hicact_speeds_hist"),verbose=True)

hicact_speeds(df_hicact_a.v, df_hicact_b.v)

# Scatterplot of STEREO-Ahead speeds against position angles
def hicact_a_speeds_pa():
	plt.scatter(df_hicact_a.v,df_hicact_a.pa,s=80,facecolor='red',edgecolor='none',alpha=0.25)
	#plt.axhline(y=df_hicact_b.pa.min())
	#plt.axhline(y=df_hicact_b.pa.max())
	plt.xlim([0,2100])
	plt.title("HICACTus STEREO-Ahead")
	plt.xlabel("Speed [kms-1]")
	plt.ylabel("Position Angle [deg]")
	save(path=os.path.join(config.hicact_path,"hicact_a_speeds_pa"),verbose=True)

hicact_a_speeds_pa()

# Scatterplot of STEREO-Behind speeds against position angles
def hicact_b_speeds_pa():
	plt.scatter(df_hicact_b.v,df_hicact_b.pa,s=80,edgecolor='none',alpha=0.25)
	#plt.axhline(y=df_hicact_b.pa.min())
	#plt.axhline(y=df_hicact_b.pa.max())
	plt.xlim([0,2100])
	plt.title("HICACTus STEREO-Behind")
	plt.xlabel("Speed [kms-1]")
	plt.ylabel("Position Angle [deg]")
	save(path=os.path.join(config.hicact_path,"hicact_b_speeds_pa"),verbose=True)

hicact_b_speeds_pa()

# Scatterplot of STEREO-Ahead & Behind speeds against position angles
def hicact_speeds_pa():
	plt.figure(num=None,figsize=(6,8),dpi=80,facecolor='w',edgecolor='k')
	a = plt.scatter(df_hicact_a.v,df_hicact_a.pa,s=20,facecolor='red',edgecolor='none',alpha=0.25)
	b = plt.scatter(df_hicact_b.v,df_hicact_b.pa,s=20,facecolor='blue',edgecolor='none',alpha=0.25)
	plt.xlim([0,2100])
	plt.ylim([0,360])
	plt.title("HICACTus STEREO CMEs")
	plt.xlabel("Speed [kms-1]")
	plt.ylabel("Position Angle [deg]")
	plt.legend([a,b],['Ahead','Behind'],prop={'size':8})
	save(path=os.path.join(config.hicact_path,"hicact_speeds_pa"),verbose=True)

hicact_speeds_pa()

'''
wp3_speeds=df_hicat[['FP speed [kms-1]','SSE speed [kms-1]','HM speed [kms-1]']]
wp3_speeds.plot(kind='hist',stacked=True,bins=100)
save(path=os.path.join(config.hicat_path,"wp3_speeds_hist"),verbose=True)
'''

# Split speeds by year
def hicact_speeds_datetime():
	import datetime
	time_format = "%Y/%m/%d %H:%M"
	datetimes_a = np.array([datetime.datetime.strptime(x,time_format) for x in df_hicact_a.starttime])
	datetimes_b = np.array([datetime.datetime.strptime(x,time_format) for x in df_hicact_b.starttime])
	plt.figure(num=None,figsize=(6,8),dpi=80,facecolor='w',edgecolor='k')
	a = plt.scatter(df_hicact_a.v,datetimes_a,s=20,facecolor='red',edgecolor='none',alpha=0.25)
	b = plt.scatter(df_hicact_b.v,datetimes_b,s=20,facecolor='blue',edgecolor='none',alpha=0.25)
	plt.xlim([0,2100])
	plt.title("HICACTus STEREO CMEs")
	plt.xlabel("Speed [kms-1]")
	plt.ylabel("Time")
	plt.legend([a,b],['Ahead','Behind'],prop={'size':8})
	save(path=os.path.join(config.hicact_path,"hicact_speeds_datetimes"),verbose=True)

hicact_speeds_datetime()

def fourier_speeds():
	from scipy.fftpack import fft
	yf = fft(df_hicact_b.v)
	plt.scatter(df_hicact_b.v,yf)
	#plt.show()

fourier_speeds()

