#!/usr/bin/env python

'''
Project	:	HELCATS

Name	:	read_cdaw

Purpose	:	Read the CDAW catalog

Explanation:	Input the CDAW catalog as a text file and generate a dataframe to house the columns of interest, output as a CSV.

Use	:	$ python read_cdaw.py
		>>> from read_cdaw import cdaw

Inputs	:	filename - of CDAW text file "univ_all.txt" sourced from http://cdaw.gsfc.nasa.gov/CME_list/UNIVERSAL/text_ver/univ_all.txt

Outputs	:	CSV file of CDAW catalog "cdaw_catalog.csv"

Keywords:	

Calls	:	config, os, numpy, matplotlib, pandas

Written	:	Jason P Byrne, STFC/RAL Space, Nov 2015 (jason.byrne@stfc.ac.uk)

Revisions:
2015-11-28 JPB : Editing the directory paths according to specified file locations in config.py	

'''

import config
import os
import numpy as np
import pandas as pd
import urllib2

def main():
	###
	# Loop over lines in text file, creating a list of dictionaries
	# Open the CDAW CME text file for reading
	f=open(os.path.join(config.cdaw_path,'univ_all.txt'),'r')
	#f = urllib2.urlread("http://cdaw.gsfc.nasa.gov/CME_list/UNIVERSAL/text_ver/univ_all.txt")
	# Read in the header lines first
	hdr1 = f.readline()
	hdr2 = f.readline()
	hdr3 = f.readline()
	hdr4 = f.readline()
	# initialise the data list
	data = []
	for line in f:
		line = line.strip()
		columns = line.split()
		source = {}
		source['date'] = columns[0]
		source['time'] = columns[1]
		source['cpa'] = int(columns[2]) if columns[2]!='Halo' else np.NaN
		source['width'] = int(columns[3])
		source['lin_speed'] = int(columns[4]) if columns[4]!='----' else np.NaN
		source['quad_speed_init'] = int(columns[5]) if columns[5]!='----' else np.NaN
		source['quad_speed_final'] = int(columns[6]) if columns[6]!='----' else np.NaN
		source['quad_speed_20'] = int(columns[7]) if columns[7]!='----' else np.NaN
		source['accel'] = float(columns[8].strip('*')) if columns[8]!='------' else np.NaN
		source['mpa'] = int(columns[11])
		if len(columns)>12:
			source['remarks'] = ' '.join(columns[12:])
		data.append(source)
	
	f.close()
	
	#Creating a pandas dataframe
	cols = ['date','time','cpa','width','lin_speed','quad_speed_init','quad_speed_final','quad_speed_20','accel','mpa','remarks']
	cdaw = pd.DataFrame(data,columns=cols)
	# Convert strings to numerics
	# cdaw = cdaw.convert_objects(convert_numeric=True)
	# Drop NaNs
	# cdaw.dropna(axis='rows',how='any',inplace=True)
	# Save out dataframe as a new CSV:
	cdaw.to_csv(os.path.join(config.cdaw_path,"cdaw_catalog.csv"),index=False)	
	

# Run main() function above to generate the CDAW dataframe, else read in previously saved CSV version.
if __name__=="__main__":
	main()
else:
	if os.path.exists(os.path.join(config.cdaw_path,"cdaw_catalog.csv")):
		cdaw = pd.read_csv(os.path.join(config.cdaw_path,"cdaw_catalog.csv"))
	else:
		cdaw = -1
		print("File Not Found: 'cdaw_catalog.csv'")
		print("^may need to run 'python read_cdaw.py' to generate it.")

