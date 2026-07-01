#!/usr/bin/env python
# -*- coding: utf-8 -*-

def Sweep(maxPower, maxSupply, horizScale, vertScale, stepValue, stepNumber, offsetValue, slowSweepBool, polarity, parent_window=None):
    print("Starting Sweep Measurement...")
    import pyvisa
    import re
    import time
    import matplotlib.pyplot as plt
    import numpy as np
    from datetime import datetime
    import csv
    from PyQt6.QtWidgets import QApplication

    # IMPORTANT: Initialize lists to prevent errors if the measurement fails early
    Xmatrix = []
    Ymatrix = []
    offsetV = 0
    stepV = 0

    try:
        rm = pyvisa.ResourceManager()
        instlist = rm.list_resources()
        inst371B = None

        for i in instlist:
            try:
                tempinst = pyvisa.ResourceManager().open_resource(i)
                if "GPIB" in i:
                    if "371B" in tempinst.query('ID?'):
                        inst371B = tempinst
                        print("371B Instrument Found!")
                        break
            except:
                pass

        if inst371B is None:
            print("Instrument not found! Please check the connection.")
            return None, None, None, None

        supplyStatus = inst371B.query('CSOut?').split()[1]
        if supplyStatus == 'CURRENT':
            print("Current Supply Enabled: Proceeding with Settings.")
        else:
            print("Collector Supply Incorrect. Please Disable 'High Voltage' and Enable 'High Current'.")
            return None, None, None, None

        inst371B.write(str('PKPower ' + maxPower))
        inst371B.write(f'CSPol {polarity}')
        inst371B.write(str('HORiz COLlect:' + horizScale))
        inst371B.write(str('VERt COLlect:' + vertScale))
        inst371B.write('STPgen OUT:ON')
        inst371B.write(str('STPgen VOLtage:' + stepValue))
        inst371B.write(str('STPgen NUMber:' + stepNumber))
        inst371B.write(str('STPgen OFFset:' + offsetValue))
        inst371B.write('STPgen MULT:OFF')
        inst371B.write('STPgen INVert:OFF')
        inst371B.write('OPC ON')
        inst371B.write('RQS OFF')

        inst371B.write(str('VCSpply ' + maxSupply))

        print('Instrument Settings Applied: ')
        print(inst371B.query('SET?'))

        if slowSweepBool is False:
            inst371B.write('MEAsure SWEep')
        elif slowSweepBool is True:
            inst371B.write('MEAsure SSWEep')

        MeasurementOn = True

        while MeasurementOn:
            time.sleep(0.5)

            # --- GUI STOP LOGIC HANDLING ---
            if parent_window is not None:
                QApplication.processEvents()  # Prevent GUI from freezing
                if parent_window.stop_measurement_flag:
                    print("Measurement interrupted by user!")
                    inst371B.write('STOP')  # Halt the instrument
                    return None, None, None, None

            eventStr = inst371B.query('EVent?')
            eventSplit = eventStr.split()
            if int(eventSplit[1]) == 751:
                MeasurementOn = False

        inst371B.write('CUr?')
        rawCurve = inst371B.read_raw()
        filterCurve = rawCurve[26:len(rawCurve) - 1]

        X = []
        Y = []

        for kk in range(int(len(filterCurve) / 4)):
            X.append(filterCurve[4 * kk] * 256 + filterCurve[4 * kk + 1])
            Y.append(filterCurve[4 * kk + 2] * 256 + filterCurve[4 * kk + 3])

        preambleRaw = inst371B.query('WFM?')
        print(preambleRaw)
        preambleSplit = preambleRaw.replace(',', '/').split('/')

        preambleKeys = ["VERT", "HORIZ", "STEP", "OFFSET"]
        preambleSecKeys = ["XOFF", "YOFF"]
        preambleDict = {}
        for preambleLine in preambleSplit:
            for keyLoop in preambleKeys:
                if keyLoop in preambleLine:
                    preambleLineSplit = preambleLine.split()
                    if len(preambleLineSplit) == 2:
                        digits = float(re.findall(r"[-+]?(?:\d*\.*\d+)", preambleLineSplit[1])[0])
                        chars = ''.join(re.findall(r"[A-Za-z][^A-Za-z]*", preambleLineSplit[1]))
                        preambleLineSplit[1] = digits
                        preambleLineSplit.append(chars)

                    preambleLineSplit[1] = float(preambleLineSplit[1])
                    if "m" in preambleLineSplit[2]:
                        preambleLineSplit.append(0.001)
                    elif "k" in preambleLineSplit[2]:
                        preambleLineSplit.append(1000)
                    elif "u" in preambleLineSplit[2]:
                        preambleLineSplit.append(0.000001)
                    else:
                        preambleLineSplit.append(1)
                    preambleDict[keyLoop] = preambleLineSplit
            for keyLoop2 in preambleSecKeys:
                if keyLoop2 in preambleLine:
                    preambleLineSplit = preambleLine.split()
                    preambleDict[keyLoop2] = float(preambleLineSplit[1])

        # --- DATA FILTERING (Removing ADC > 4096 Overrange Errors) ---
        X_array = np.array(X)
        Y_array = np.array(Y)

        valid_mask = (X_array < 4096) & (Y_array < 4096)
        X_clean = X_array[valid_mask]
        Y_clean = Y_array[valid_mask]

        Xnorm = (X_clean - preambleDict["XOFF"]) * preambleDict["HORIZ"][1] * preambleDict["HORIZ"][3] * 10 / 1000
        Ynorm = (Y_clean - preambleDict["YOFF"]) * preambleDict["VERT"][1] * preambleDict["VERT"][3] * 10 / 1000

        XCheck = -10000
        Xtemp = []
        Ytemp = []

        for ii in range(len(Xnorm)):
            if Xnorm[ii] < XCheck - 0.2:
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

        # Plot the final graph
        plt.figure()
        plt.plot(np.array(Xmatrix).T, np.array(Ymatrix).T)
        plt.xlabel("Drain Voltage [V]")
        plt.ylabel("Source Current [A]")

        legendList = []
        offsetV = preambleDict["OFFSET"][1] * preambleDict["OFFSET"][3]
        stepV = preambleDict["STEP"][1] * preambleDict["STEP"][3]

        for mm in range(len(Xmatrix)):
            legendList.append(str(offsetV + mm * stepV) + " V")

        plt.legend(legendList, title="Gate Voltage")
        plt.savefig("Fig/Fig_" + datetime.today().strftime('%Y%m%d_%H%M%S') + ".png")

        # --- LINIA ADAUGATA AICI ---
        plt.show()

        print("Sweep Measurement Finished Successfully.")

        # Return the processed data to the GUI
        return Xmatrix, Ymatrix, offsetV, stepV

    except Exception as e:
        print(f"Error occurred during measurement: {e}")
        return None, None, None, None