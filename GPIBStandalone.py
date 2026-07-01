#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pyvisa
import re
import time
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

rm = pyvisa.ResourceManager( )
instlist = rm.list_resources()

for i in instlist:
	tempinst = pyvisa.ResourceManager().open_resource(i)
	if "371B" in tempinst.query('ID?'):
		inst371B = tempinst
		print("371B Found")

#inst371B.write('TEXt "ciao"')

inst371B.write('PKPower 300') # it's the maximum peak power in W for the interlock (0.03, 0.3, 3, 30, 300, 3000). Only 300 or 3000W for the high current mode
inst371B.write('VCSpply 100.0') # this is the highest voltage applied, as a percentage of the highest possible (30 V in high voltage mode)

inst371B.write('OPC ON') #needed to get the event information, to know when a sweep is finished
inst371B.write('RQS OFF') # I had some problems with the event ID when ON. I believe it's related to printing
inst371B.write('MEAsure SWEep')
time.sleep(1) # I wait in order to skip the first events, which look like the end of a sweep

MeasurementOn = True

while MeasurementOn:
	time.sleep(1)
	eventStr = inst371B.query('EVent?')
	eventSplit = eventStr.split()
	if int(eventSplit[1]) > 0: # To be checked
		MeasurementOn = False

inst371B.write('CUr?')
rawCurve = inst371B.read_raw()

filterCurve = rawCurve[26:len(rawCurve)-1] # Removing header and isolate the data (the last byte is just a checksum)

#print(filterCurve)

X = []
Y = []

# horrible decoding. The data is divided in XXYYXXYY..., so 2 bytes for an X-value followed by 2 bytes for the corresponding Y-value
for kk in range(int(len(filterCurve)/4)):
	X.append(filterCurve[4 * kk] * 256 + filterCurve[4 * kk + 1])
	Y.append(filterCurve[4 * kk + 2] * 256 + filterCurve[4 * kk + 3])

#print(X)
#print(Y)

#plt.scatter(X,Y)
#plt.show()

# Preamble decoding

preambleRaw = inst371B.query('WFM?') # WFM is the preamble
print(preambleRaw)
preambleSplit = preambleRaw.split('/')

#horrible loop to parse the meaningful quantities. It should be doable much better. The dictionary entry for a certain variable has an array with 3 entries: the name of the variable (to be removed?), numerical value, and units.
preambleKeys = ["VERT", "HORIZ", "STEP", "OFFSET"]
preambleDict = {}

for preambleLine in preambleSplit:
	#print(preambleLine)
	for keyLoop in preambleKeys:
		if keyLoop in preambleLine:
			preambleLineSplit = preambleLine.split()
			if len(preambleLineSplit) == 2: #in case of mA and mV the digits and characters are merged in a single entry
				digits = float(re.findall(r"[-+]?(?:\d*\.*\d+)", preambleLineSplit[1])[0])
				chars = ''.join( re.findall(r"[A-Za-z][^A-Za-z]*", preambleLineSplit[1]) )
				preambleLineSplit[1] = digits
				preambleLineSplit.append(chars)

			preambleLineSplit[1] = float(preambleLineSplit[1]) # I want to avoid numbers to be stored as string
			if "m" in preambleLineSplit[2]:
				preambleLineSplit.append(0.001) # I could replace the unit string, but I keep it for debugging
			elif "k" in preambleLineSplit[2]:
				preambleLineSplit.append(1000)
			elif "u" in preambleLineSplit[2]:
				preambleLineSplit.append(0.000001)
			else:
				preambleLineSplit.append(1)
			preambleDict[keyLoop] = preambleLineSplit	
#print(preambleDict)

# Normalization. The raw values are multiplied by 10 times the value of a quadrant and divided by 1023, the maximum value
Xnorm = np.array(X) * preambleDict["HORIZ"][1] * preambleDict["HORIZ"][3] * 10 / 1023
Ynorm = np.array(Y) * preambleDict["VERT"][1] * preambleDict["VERT"][3] * 10 / 1023

#print(Xnorm)
#print(Ynorm)

print(inst371B.query('SET?'))

#inst371B.write('PLOt CURve')
#print(inst371B.read_raw())

# This is to divide the curves
Xmatrix = []
Ymatrix = []

XCheck = -10000
Xtemp = []
Ytemp = []

for ii in range(len(Xnorm)):
	if Xnorm[ii] < XCheck - 0.2: #the 0.2 is to avoid some edge cases where somehow the voltage decreased randomly. it should be removed
		XCheck = -10000
		Xmatrix.append(Xtemp)
		Ymatrix.append(Ytemp)
		Xtemp = []
		Ytemp = []
	Xtemp.append(Xnorm[ii])
	Ytemp.append(Ynorm[ii])
	XCheck = Xnorm[ii]

Xmatrix.append(Xtemp)
Ymatrix.append(Ytemp)
#print(Xmatrix)

plt.plot(np.array(Xmatrix).T,np.array(Ymatrix).T)
plt.xlabel("Drain Voltage [V]")
plt.ylabel("Source Current [A]")

# Plot legend
legendList = []
offsetV = preambleDict["OFFSET"][1] * preambleDict["OFFSET"][3]
stepV = preambleDict["STEP"][1] * preambleDict["STEP"][3]

for mm in range(len(Xmatrix)):
	legendList.append(str(offsetV + mm * stepV) + " V")
		
plt.legend(legendList, title = "Gate Voltage")

plt.savefig("Fig/Fig" + datetime.today().strftime('%Y%m%d%H%M%S') + ".png")
plt.show()