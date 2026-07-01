#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import csv
import numpy as np

from PyQt6 import QtWidgets as QtW
from PyQt6 import QtCore

import configparser

from Sweep371B import Sweep
from CheckConfig371B import CheckConfig


class MainWindow(QtW.QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Tektronix 371B GPIB Controller")

        # --- DATA AND CONTROL VARIABLES ---
        self.stop_measurement_flag = False
        self.last_x_matrix = None
        self.last_y_matrix = None
        self.last_offset = None
        self.last_step = None

        layout = QtW.QFormLayout(self)
        self.setLayout(layout)

        self.settings2 = QtCore.QSettings('Tektronix 371B', 'settings2')

        self.loadFileButton = QtW.QPushButton("Open Config File")
        self.loadFileButton.clicked.connect(self.loadConfigfile)
        layout.addRow("Load Config File", self.loadFileButton)

        # --- POLARITY SETTING ---
        self.polarityBox = QtW.QComboBox()
        self.polarityBox.addItem("NPN / N-Channel", "NPN")
        self.polarityBox.addItem("PNP / P-Channel", "PNP")
        layout.addRow("Device Polarity", self.polarityBox)

        self.powerBox = QtW.QComboBox()
        self.powerBox.addItem("300 W", "300")
        self.powerBox.addItem("3 kW", "3000")
        layout.addRow("Maximum Power", self.powerBox)

        self.maxSupplyBox = QtW.QLineEdit("30")
        layout.addRow("Maximum Collector Voltage [V]", self.maxSupplyBox)

        self.HorizBox = QtW.QComboBox()
        self.HorizBox.addItem("100 mV/div", "0.1")
        self.HorizBox.addItem("200 mV/div", "0.2")
        self.HorizBox.addItem("500 mV/div", "0.5")
        self.HorizBox.addItem("1 V/div", "1")
        self.HorizBox.addItem("2 V/div", "2")
        self.HorizBox.addItem("5 V/div", "5")
        layout.addRow("Horizontal Scale", self.HorizBox)

        self.VertBox = QtW.QComboBox()
        self.VertBox.addItem("500 mA/div", "0.5")
        self.VertBox.addItem("1 A/div", "1")
        self.VertBox.addItem("2 A/div", "2")
        self.VertBox.addItem("5 A/div", "5")
        self.VertBox.addItem("10 A/div", "10")
        self.VertBox.addItem("20 A/div", "20")
        self.VertBox.addItem("50 A/div", "50")
        layout.addRow("Vertical Scale", self.VertBox)

        self.OffsetBox = QtW.QLineEdit("2")
        layout.addRow("Gate Voltage Start Value [V]", self.OffsetBox)

        self.StepBox = QtW.QComboBox()
        self.StepBox.addItem("200 mV", "0.2")
        self.StepBox.addItem("500 mV", "0.5")
        self.StepBox.addItem("1 V", "1")
        self.StepBox.addItem("2 V", "2")
        self.StepBox.addItem("5 V", "5")
        layout.addRow("Gate Voltage Step", self.StepBox)

        self.StepNumberBox = QtW.QComboBox()
        self.StepNumberBox.addItems(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"])
        layout.addRow("Number of Steps", self.StepNumberBox)

        self.sweepBox = QtW.QCheckBox(text="Slow Sweep")
        layout.addWidget(self.sweepBox)

        button = QtW.QPushButton('Save Configuration')
        button.clicked.connect(self.saveConfigfile)
        layout.addWidget(button)

        # --- NEW ACTION BUTTONS ---

        # Horizontal layout for Start/Stop
        btn_layout = QtW.QHBoxLayout()

        self.btn_start = QtW.QPushButton('Start Measurement')
        self.btn_start.clicked.connect(self.StartMeasurement)
        self.btn_start.setStyleSheet("background-color: lightgreen; font-weight: bold;")
        btn_layout.addWidget(self.btn_start)

        self.btn_stop = QtW.QPushButton('Stop Measurement')
        self.btn_stop.clicked.connect(self.StopMeasurement)
        self.btn_stop.setStyleSheet("background-color: salmon; font-weight: bold;")
        self.btn_stop.setEnabled(False)  # Disabled until measurement starts
        btn_layout.addWidget(self.btn_stop)

        layout.addRow(btn_layout)

        # Export Button
        self.btn_export = QtW.QPushButton('Export to CSV')
        self.btn_export.clicked.connect(self.ExportCSV)
        self.btn_export.setStyleSheet("background-color: lightblue;")
        layout.addWidget(self.btn_export)

        self.setFixedWidth(400)
        self.setFixedHeight(450)

        # Added 'polarity' to WidgetMap so it remembers your NPN/PNP choice when saving configs
        self.WidgetMap = {
            'polarity': self.polarityBox,
            'maxpower': self.powerBox,
            'maxvoltage': self.maxSupplyBox,
            'horizscale': self.HorizBox,
            'vertscale': self.VertBox,
            'gateoffset': self.OffsetBox,
            'gatestep': self.StepBox,
            'stepnumber': self.StepNumberBox,
            'slowsweep': self.sweepBox
        }

        self.show()

    def loadConfigfile(self):
        loadConfigPath = QtW.QFileDialog.getOpenFileName(self, 'Open Config File',
                                                         str(os.path.dirname(__file__) + "/Configs"))
        print("Loading File: " + str(loadConfigPath[0]))
        loadconfigParser = configparser.RawConfigParser()
        loadconfigParser.read(loadConfigPath[0])

        for name, widget in self.WidgetMap.items():
            cls = widget.__class__.__name__
            if cls == "QCheckBox":
                self.WidgetMap[name].setChecked(loadconfigParser.get('configuration', name) == "True")
            elif cls == "QLineEdit":
                self.WidgetMap[name].setText(loadconfigParser.get('configuration', name))
            elif cls == "QComboBox" and name == "stepnumber":
                self.WidgetMap[name].setCurrentText(loadconfigParser.get('configuration', name))
            elif cls == "QComboBox":
                index = self.WidgetMap[name].findData(str(loadconfigParser.get('configuration', name)))
                self.WidgetMap[name].setCurrentIndex(index)
            else:
                print("Widget not recognized.")

        print("Configuration loaded successfully.")

    def saveConfigfile(self):
        config = configparser.ConfigParser()
        config.add_section('configuration')

        for name, widget in self.WidgetMap.items():
            cls = widget.__class__.__name__
            if cls == "QCheckBox":
                config['configuration'][name] = str(self.WidgetMap[name].isChecked())
            elif cls == "QLineEdit":
                config['configuration'][name] = self.WidgetMap[name].text()
            elif cls == "QComboBox" and name == "stepnumber":
                config['configuration'][name] = self.WidgetMap[name].currentText()
            elif cls == "QComboBox":
                config['configuration'][name] = self.WidgetMap[name].currentData()
            else:
                print("Widget not recognized.")

        saveConfigPath = QtW.QFileDialog.getSaveFileName(self, 'New Config File Name',
                                                         str(os.path.dirname(__file__) + "/Configs"))
        with open(saveConfigPath[0], 'w') as configfile:
            config.write(configfile)

        print("Configuration saved successfully.")

    def GetSettings(self):
        polaritySetting = self.polarityBox.currentData()  # Extract NPN or PNP
        powerSetting = self.powerBox.currentData()
        maxSupplySetting = self.maxSupplyBox.text()
        horizSetting = self.HorizBox.currentData()
        vertSetting = self.VertBox.currentData()
        offsetSetting = self.OffsetBox.text()
        stepSetting = self.StepBox.currentData()
        stepNumberSetting = self.StepNumberBox.currentText()
        slowSweep = self.sweepBox.isChecked()

        # Convert maximum applied voltage in percentage of maximum voltage (30 V)
        maxSupplyConverted = str(float(maxSupplySetting) * 100. / 30.)

        # Convert offset in units of step
        offsetConverted = str(float(offsetSetting) / float(stepSetting))

        # We now return 9 elements instead of 8 (polarity is settings[0])
        return polaritySetting, powerSetting, maxSupplyConverted, horizSetting, vertSetting, stepSetting, stepNumberSetting, offsetConverted, slowSweep

    # --- ACTION BUTTONS LOGIC ---

    def StopMeasurement(self):
        self.stop_measurement_flag = True

    def StartMeasurement(self):
        self.stop_measurement_flag = False
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)

        settings = self.GetSettings()

        # Pass 'self' as parent_window so Sweep371B can read the stop_measurement_flag
        # parameters are now: power(1), supply(2), horiz(3), vert(4), step(5), step_num(6), offset(7), slow_sweep(8), polarity(0)
        X_res, Y_res, offset, step = Sweep(settings[1], settings[2], settings[3], settings[4], settings[5], settings[6],
                                           settings[7], settings[8], settings[0], self)

        # If measurement succeeded, store the results in memory for exporting
        if X_res is not None:
            self.last_x_matrix = X_res
            self.last_y_matrix = Y_res
            self.last_offset = offset
            self.last_step = step
            print("Data stored in memory. Click 'Export to CSV' to save the file.")

        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def ExportCSV(self):
        if self.last_x_matrix is None or self.last_y_matrix is None:
            QtW.QMessageBox.warning(self, "Error", "No measurement data available! Please run a measurement first.")
            return

        save_path, _ = QtW.QFileDialog.getSaveFileName(self, 'Save CSV Measurement', '', 'CSV Files (*.csv)')

        if save_path:
            OutputMatrix = []
            for mm in range(len(self.last_y_matrix)):
                self.last_x_matrix[mm].insert(0, "Gate Voltage")
                self.last_y_matrix[mm].insert(0, self.last_offset + mm * self.last_step)
                OutputMatrix.append(self.last_x_matrix[mm])
                OutputMatrix.append(self.last_y_matrix[mm])

            # Export in a standard Excel-friendly CSV format
            with open(save_path, 'w', newline='') as f:
                writer = csv.writer(f)
                # Transpose matrix to write vertical columns
                for row in zip(*OutputMatrix):
                    writer.writerow(row)

            QtW.QMessageBox.information(self, "Success", f"File successfully exported to:\n{save_path}")


if __name__ == "__main__":
    app = QtW.QApplication([])
    window = MainWindow()
    sys.exit(app.exec())