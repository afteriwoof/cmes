#!/usr/bin/env python

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import config

from savefig import save

from cmes import cdaw,hicactus,hicat

# Call the catalog functions:
df_cdaw = cdaw()
df_hicact_a = hicactus('A')
df_hicact_b = hicactus('B')
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

v_a = np.array(df_hicact_a[['v']].astype('float'))
plt.hist(v_a,bins=np.arange(0, max(v_a) + binwidth, binwidth))
plt.title("HICACTus STEREO-Ahead")
save(path=os.path.join(config.hicact_path,"hicact_a_speeds_hist"),verbose=True)

v_b = np.array(df_hicact_b[['v']].astype('float'))
plt.hist(v_b,bins=np.arange(0,max(v_b)+binwidth,binwidth))
plt.title("HICACTus STEREO-Behind")
save(path=os.path.join(config.hicact_path,"hicact_b_speeds_hist"),verbose=True)

plt.hist(v_a,bins=np.arange(0,max(v_a)+binwidth,binwidth),histtype='stepfilled',\
	normed=False,color='b',label='Ahead')
plt.hist(v_b,bins=np.arange(0,max(v_b)+binwidth,binwidth),histtype='stepfilled',\
	normed=False,color='r',alpha=0.5,label='Behind')
plt.title("HICACTus CME Speeds")
plt.xlabel("Speed [kms-1]")
plt.ylabel("Count")
plt.legend(prop={'size':8})
save(path=os.path.join(config.hicact_path,"hicact_speeds_hist"),verbose=True)


wp3_speeds=df_hicat[['FP speed [kms-1]','SSE speed [kms-1]','HM speed [kms-1]']]
wp3_speeds.plot(kind='hist',stacked=True,bins=100)
save(path=os.path.join(config.hicat_path,"wp3_speeds_hist"),verbose=True)


# Split speeds by year


