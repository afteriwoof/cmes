
'''
Project	:	HELCATS
Name	:	cmes
Purpose	:	Read the CME catalog
Explanation:	Input the CDAW CME catalog as a text file and generate a dataframe to house the columns of interest, output as a CSV.
Use	:	$ python cmes.py
Inputs	:	filename - of CDAW text file "univ_all.txt" sourced from http://cdaw.gsfc.nasa.gov/CME_list/UNIVERSAL/text_ver/univ_all.txt
Outputs	:	CSV file of CDAW catalog "cdaw_catalog.csv"
Keywords:	
Calls	:	os, numpy, matplotlib, pandas
Written	:	Jason P Byrne, STFC/RAL Space, Dec 2015 (jason.byrne@stfc.ac.uk)
Revisions:
2015-12-01 JPB : Working in accordance with sunpy project guidelines.
2015-12-03 JPB	:	Adding hicactus function.
'''

import os
import numpy as np
import pandas as pd
import urllib2
import config

def cdaw():
	
	try:
		#f = urllib2.urlopen("http://cdaw.gsfc.nasa.gov/CME_list/UNIVERSAL/text_ver/univ_all.txt")
		# Open the CDAW CME text file for reading
		f = open(os.path.join(config.cdaw_path,'univ_all.txt'),'r') 
		# Read in the header lines first
		for i in range(4):
			f.readline()
		
		# initialise the data list
		cols = ['date','time','cpa','width','lin_speed','quad_speed_init',\
			'quad_speed_final','quad_speed_20','accel','mass','kin_energy','mpa','remarks']
		source = {k:list() for k in cols}
		for line in f:
			
			line = line.replace('Halo','360').strip()
			columns = line.split()
			columns = [col.strip('*') if ("--" not in col) and ("**" not in col) else np.NaN for col in columns]
			if len(columns)<13:
				columns.append(' ')
			for i,(k,val) in enumerate(zip(cols,columns)):
				if i==12:
					source[k].append(' '.join(columns[i:]))
				else:
					source[k].append(val)
				
		f.close()
	

		#Creating a pandas dataframe
		return pd.DataFrame(source,columns=cols)

	except:
		raise

# spc = 'A' or 'B' for the Ahead or Behind spacecraft
def hicactus(spc):
	###
	# Read in the HELCATS HI CACTUS CME list
	try:
		spc.upper() in ('A','B')
		#f = urllib2.urlopen("http://solwww.oma.be/users/sarahw/hicactus/catalog/A/catacmeA.txt")
		fl = "".join(('catacme',spc,'.txt'))
		f = open(os.path.join(config.hicact_path,fl),'r')

		# Read in the header lines first
		hdr = f.readline()
		
		# Initialise the data list
		cols = [x.strip() for x in hdr.rstrip().split('|')]
		source = {k:list() for k in cols}

		for line in f:
			columns = [x.strip() for x in line.rstrip().split('|')]

			for k,val in zip(cols,columns):
				source[k].append(val)
	
		f.close()

		# Create a pandas dataframe
		return pd.DataFrame(source,columns=cols)

	except:
		if spc.upper() not in ('A','B'): print("Input must be 'A' or 'B' to indicate STEREO-Ahead or Behind")

		raise

def hicat():
	# Read in the HELCATS HI Observational CME list
	try:
		#f = urllib2.urlopen("http://www.helcats-fp7.eu/catalogues/data/HCME_WP3_V02.txt")
		#f = open(os.path.join(config.hicat_path,'HCME_WP3_V02.txt'),'r')

		cols = ['ID','Date [UTC]','SC','L-N','PA-N [deg]','L-S','PA-S [deg]','Quality','PA-fit',\
			'FP speed [kms-1]','FP speed Err [kms-1]','FP Phi [deg]','FP Phi Err [deg]',\
			'FP HEEQ Long [deg]','FP HEEQ Lat [deg]','FP Carr Long [deg]','FP Launch [UTC]',\
			'SSE speed [kms-1]','SSE speed Err [kms-1]','SSE Phi [deg]','SSE Phi Err [deg]',\
			'SSE HEEQ Long [deg]','SSE HEEQ Lat [deg]','SSE Carr Long [deg]','SSE Launch [UTC]',\
			'HM speed [kms-1]','HM speed Err [kms-1]','HM Phi [deg]','HM Phi Err [deg]',\
                        'HM HEEQ Long [deg]','HM HEEQ Lat [deg]','HM Carr Long [deg]','HM Launch [UTC]']

		fl_format = 'votable'
		from astropy.table import Table
		t = Table.read(os.path.join(config.hicat_path,'HCME_WP3_V02.vot'),format=fl_format)
		t = Table(t,names=cols)
		return pd.DataFrame(np.array(t),columns=cols)

	except:
		raise

