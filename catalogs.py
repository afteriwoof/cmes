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


hicact_a_speeds = df_hicact_a[['v']]
hicact_b_speeds = df_hicact_b[['v']]

binwidth = 50
# Histogram of STEREO-Ahead speeds
v_a = np.array(df_hicact_a[['v']].astype('float'))
plt.hist(v_a,bins=np.arange(0, max(v_a) + binwidth, binwidth))
plt.title("HICACTus STEREO-Ahead")
save(path=os.path.join(config.hicact_path,"hicact_a_speeds_hist"),verbose=True)

# Histogram of STEREO-Behind speeds
v_b = np.array(df_hicact_b[['v']].astype('float'))
plt.hist(v_b,bins=np.arange(0,max(v_b)+binwidth,binwidth))
plt.title("HICACTus STEREO-Behind")
save(path=os.path.join(config.hicact_path,"hicact_b_speeds_hist"),verbose=True)

# Histogram of STEREO-Ahead & Behind speeds
plt.hist(v_a,bins=np.arange(0,max(v_a)+binwidth,binwidth),histtype='stepfilled',\
	normed=False,color='b',label='Ahead')
plt.hist(v_b,bins=np.arange(0,max(v_b)+binwidth,binwidth),histtype='stepfilled',\
	normed=False,color='r',alpha=0.5,label='Behind')
plt.title("HICACTus CME Speeds")
plt.xlabel("Speed [kms-1]")
plt.ylabel("Count")
plt.legend(prop={'size':8})
save(path=os.path.join(config.hicact_path,"hicact_speeds_hist"),verbose=True)

# Scatterplot of STEREO-Ahead speeds against position angles
plt.scatter(df_hicact_a.v,df_hicact_a.pa,s=80,facecolor='red',edgecolor='none',alpha=0.25)
#plt.axhline(y=df_hicact_b.pa.min())
#plt.axhline(y=df_hicact_b.pa.max())
plt.xlim([0,2100])
plt.title("HICACTus STEREO-Ahead")
plt.xlabel("Speed [kms-1]")
plt.ylabel("Position Angle [deg]")
save(path=os.path.join(config.hicact_path,"hicact_a_speeds_pa"),verbose=True)

# Scatterplot of STEREO-Behind speeds against position angles
plt.scatter(df_hicact_b.v,df_hicact_b.pa,s=80,edgecolor='none',alpha=0.25)
#plt.axhline(y=df_hicact_b.pa.min())
#plt.axhline(y=df_hicact_b.pa.max())
plt.xlim([0,2100])
plt.title("HICACTus STEREO-Behind")
plt.xlabel("Speed [kms-1]")
plt.ylabel("Position Angle [deg]")
save(path=os.path.join(config.hicact_path,"hicact_b_speeds_pa"),verbose=True)

# Scatterplot of STEREO-Ahead & Behind speeds against position angles
plt.figure(num=None,figsize=(6,8),dpi=80,facecolor='w',edgecolor='k')
a,=plt.scatter(df_hicact_a.v,df_hicact_a.pa,s=20,facecolor='red',edgecolor='none',alpha=0.25)
b,=plt.scatter(df_hicact_b.v,df_hicact_b.pa,s=20,facecolor='blue',edgecolor='none',alpha=0.25)
plt.xlim([0,2100])
plt.ylim([0,360])
plt.title("HICACTus STEREO CMEs")
plt.xlabel("Speed [kms-1]")
plt.ylabel("Position Angle [deg]")
plt.legend([a,b],['Ahead','Behind'],prop={'size':8})
save(path=os.path.join(config.hicact_path,"hicact_speeds_pa"),verbose=True)
# Histogram of STEREO-Behind speeds split by position angle ranges
#def pa_hist(df,pa1,pa2):
#	df = df[(df.pa >= pa1) & (df.pa < pa2)]
#	plt.hist(np.array(df.v),bins=np.arange(0,max(df.v)+binwidth,binwidth))
#	plt.title("HICACTus STEREO-Behind : PA %d-%d" %(pa1,pa2))
#	plt.show()
#
#pa1=225
#n=5
#pa2=pa1+n
#for i in np.arange(100/n):
#	pa1 += n
#	pa2 += n
#	pa_hist(df_hicact_b,pa1,pa2)




'''
wp3_speeds=df_hicat[['FP speed [kms-1]','SSE speed [kms-1]','HM speed [kms-1]']]
wp3_speeds.plot(kind='hist',stacked=True,bins=100)
save(path=os.path.join(config.hicat_path,"wp3_speeds_hist"),verbose=True)
'''

# Split speeds by year


