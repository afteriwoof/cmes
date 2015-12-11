
'''
Project :       HELCATS
Name    :       compare
Purpose :       Compare the HICAT and CDAW catalogs
Explanation:   	Since HICAT is out in the heliosphere and at different observational vantage points based on where the STEREO spacecraft are at a given time, it is difficult to compare to current coronagraph catalogs, but we will try!
Use     :       $ python compare.py
                > import compare
                > compare.run_all()
Inputs  :       Provided by the cmes.py module.
Outputs :       png files
Keywords:       
Calls   :       os, numpy, matplotlib, pandas
                config, savefig, cmes
Written :       Jason P Byrne, STFC/RAL Space, Dec 2015 (jason.byrne@stfc.ac.uk)
Revisions:
2015-12-11 JPB : 
'''

import os
import numpy as np
import matplotlib.pyplot as plt
import config
from savefig import save
from cmes import cdaw, hicat

df_cdaw = cdaw().convert_objects(convert_numeric=True)
df_hicat = hicat().convert_objects(convert_numeric=True)

#global variables
speeds_label = "Speed ($km s^{-1}$)"
ledge_sz = 10

cols_hist = ['cpa','mpa','width','lin_speed','quad_speed_init','quad_speed_20',\
        'quad_speed_final','accel']

# Split speeds by year
def speeds_datetime():
        import datetime
        import matplotlib.ticker as ticker
        time_format_cdaw = "%Y/%m/%dT%H:%M:%S"
        time_format_hicat = "%Y-%m-%dT%H:%MZ"
	datetimes_cdaw = np.array([datetime.datetime.strptime(x,time_format_cdaw) \
                for x in df_cdaw.date.values+'T'+df_cdaw.time.values])
        datetimes_hicat = np.array([datetime.datetime.strptime(x[0],time_format_hicat) \
		for x in df_hicat[['Date [UTC]']].values])
	
	fig = plt.figure(num=None,figsize=(10,7),dpi=80,facecolor='w',edgecolor='k')
        ax = fig.add_subplot(111)
	ssz=10
        #lin = plt.scatter(datetimes,df_cdaw.lin_speed,s=ssz,facecolor='red',\
        #       edgecolor='none',alpha=0.3)
        q_f = plt.scatter(datetimes_cdaw,df_cdaw.quad_speed_final,s=ssz,facecolor='blue',\
                edgecolor='none',alpha=0.3)
	hi = plt.scatter(datetimes_hicat,df_hicat[['FP speed [kms-1]']],s=ssz,\
		facecolor='red',edgecolor='none',alpha=0.3)
        plt.ylim([0,3500])
        plt.title("Comparing HICAT with the CDAW LASCO CME Catalog")
        plt.ylabel(speeds_label)
        plt.xlabel("Time")
        #ax = plt.axes()
        #ax.xaxis.set_major_locator(ticker.MultipleLocator(365))
        #labels=ax.get_xticklabels()
        #plt.setp(labels,rotation=40)
        #plt.legend([q_f,hi],['CDAW','HICAT'],prop={'size':ledge_sz})
        #plt.legend([lin,q_f],['Linear','Quad. (final)'],prop={'size':ledge_sz})
        save(path=os.path.join(config.hicat_path,"compare_speeds_datetimes"),verbose=True)

speeds_datetime()



