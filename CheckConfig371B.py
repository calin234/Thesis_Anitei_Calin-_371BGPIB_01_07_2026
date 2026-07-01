#!/usr/bin/env python
# -*- coding: utf-8 -*-

def CheckConfig( maxPower , maxSupply, horizScale, vertScale, stepValue, stepNumber, offsetValue, slowSweepBool):
	# TO BE IMPROVED. The idea is to check the settings in order to throw an error if the tracer doesn't like some combination
	print("Starting Checking Configuration")
	
	if maxPower == "3000" and (float(vertScale) < 1 or float(vertScale) > 50):
		print("Error")
	elif maxPower == "300" and (float(vertScale) < 0.5 or float(vertScale) > 5):
		print("Error")

	if stepValue == "5" and float(stepNumber) > 5:
		print("Error")
 
	if float(offsetValue) > 10: #the offset has already been converted in units of steps
		print("Error")
	elif stepValue == "5" and float(offsetValue) > 5:
		print("Error")
	
	if float(maxSupply) > 100:
		print("Error")

	print("Checking Complete")

	return 1