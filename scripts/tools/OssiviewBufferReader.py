# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 10:41:50 2019

@author: SN-593
"""

import numpy as np
import yaml
from scripts.tools import OssiviewDataHeader
from scripts.tools.OssiviewDataHeader import OssiviewDataHeader

class OssiviewBufferReader:
    def __init__(self, filePath):
        self.filePath = filePath
        self.header = OssiviewDataHeader(filePath)
        self.metaData = self.header.metaData['Header']
        if self.metaData['Header Version'] == 'V1':
            self.data = self._getDataV1()
        else:
            self.data = self._getDataV2()

    def _getDataV2(self):

        # if not copy == True:
        with open(self.filePath, 'rb') as f:
            # read through the header
            f.read(self.header.fullLen)
            # Data = {}
            for Buffer in self.metaData["DataParameters"]:
                bufferType = self.getTypeMap()[Buffer['DataType']]
                Dim = Buffer["Dim"]
                if (Buffer["DataType"] == "struct DopplerData"):
                    Dim["Z"] = 2 * Dim["Z"]
                elif (Buffer["DataType"] == "struct __half2"):

                    Dim["Z"] = 2 * Dim["Z"]
                    # else:
                    #     Dim["Z"] = Dim["Z"]

                dataLength = Dim["N"] * Dim['Z'] * Dim['X'] * Dim['Y']
                dat = np.fromfile(f, dtype=bufferType, count=dataLength)
                shape = (Dim["N"], Dim['Y'], Dim['X'], Dim['Z'])
                dat = np.reshape(dat, shape)

            return dat

    def _getDataV1(self):
        with open(self.filePath, 'rb') as f:
            # read through the header
            f.read(self.header.fullLen)
            Data = {}
            for Buffer in self.metaData["Buffers"]:
                bufferType = self.getTypeMap()[Buffer['Data Type']]
                Dim = Buffer["Dim"]
                if (Buffer["Data Type"] == "struct DopplerData"):
                    Dim["Z"] = 2 * Dim["Z"]
                dataLength = Dim["N"] * Dim['Z'] * Dim['X'] * Dim['Y']
                dat = np.fromfile(f, dtype=bufferType, count=dataLength)
                shape = (Dim["N"], Dim['Y'], Dim['X'], Dim['Z'])
                dat = np.reshape(dat, shape)
                # reshuffle the entries to make the mask and doppler their own elements
                if (Buffer["Data Type"] == "struct DopplerData"):
                    doppler = dat[0][0][0][0::2][::-1]
                    mask = dat[0][0][0][1::2][::-1]
                    Data[Buffer["Common Name"]] = {"mask": mask,
                                                   "doppler": doppler}
                else:
                    Data[Buffer["Common Name"]] = dat
            return Data

    def getTypeMap(self):
        return {'struct float2': np.complex64,
                'struct DopplerData': np.complex64,
                #                'unsigned short' : np.uint16,
                'unsigned short': np.int16,
                'float': np.float32,
                'struct __half2': np.float16}

    def udpateStructureName(self, name):
        self.header.metaData['Header']['Session']['Structure'] = name

    def udpateEndoscopeFname(self, name):
        self.header.metaData['Header']['Session']['endosopePngPath'] = name

    def udpatePngFname(self, name):
        self.header.metaData['Header']['Session']['pngPath'] = name

    def updateHeader(self, Header):
        self.header.metaData = Header

    def updateData(self, data, commonName):
        self.data[commonName] = data

    def updateDataV2(self, data):
        self.data = data.astype(np.float16)

    def export(self, filePath):
        # Regenerate the buffer params
        # Merge the doppler readings together to make a list of np.array
        # instead of np.array and dict of np.array
        if "Doppler Buffer" in self.data.keys():
            maskData = self.data["Doppler Buffer"]["mask"]
            dopplerData = self.data["Doppler Buffer"]["doppler"]
            dat = np.empty((np.size(maskData) + np.size(dopplerData)), dtype=dopplerData.dtype)
            dat[0::2] = dopplerData[::-1]
            dat[1::2] = maskData[::-1]
            self.data["Doppler Buffer"] = np.reshape(dat, (1, 1, 1, np.size(dat)))

        newBuffersHeader = []
        for key, buf in self.data.items():
            bufferHeader = {}
            for typeStr, npType in self.getTypeMap().items():  # C++ type with nptype value
                if npType == buf.dtype.type:
                    if typeStr != 'struct DopplerData':
                        bufferHeader["Data Type"] = typeStr

            if key == "Doppler Buffer":
                bufferHeader["Data Type"] = 'struct DopplerData'

            bufferHeader["Common Name"] = key

            # loop through the old header to find the buffer id's we shouldn't
            # be adding buffers or changing the order of them
            for buffer in self.metaData["Buffers"]:
                if key == buffer["Common Name"]:
                    bufferHeader["Buffer ID"] = buffer["Buffer ID"]

            Dim = {}
            Dim["N"] = np.shape(buf)[0]
            Dim["Y"] = np.shape(buf)[1]
            Dim["X"] = np.shape(buf)[2]
            Dim["Z"] = np.shape(buf)[3]
            if key == "Doppler Buffer":
                Dim["Z"] = int(Dim["Z"] / 2)
            bufferHeader["Dim"] = Dim

            newBuffersHeader.append(bufferHeader)

        self.header.metaData["Header"]["Buffers"] = newBuffersHeader
        headerPrint = yaml.dump(self.header.metaData)
        headerLength = len(headerPrint.encode('ascii'))
        exportFile = open(filePath, "wb")
        exportFile.write(f"{headerLength}".encode('ascii'))
        exportFile.write(headerPrint.encode('ascii'))
        for arr in self.data:
            exportFile.write(self.data[arr].tobytes())

        exportFile.close()

        # Update the file to use the updated file
        self.filePath = filePath
        self.header = OssiviewDataHeader(filePath)
        self.data = self.getData()

    def exportV2(self, filePath):

        buf = self.data
        Dim = {}
        Dim["N"] = np.shape(buf)[0]
        Dim["Y"] = np.shape(buf)[1]
        Dim["X"] = np.shape(buf)[2]
        Dim["Z"] = np.shape(buf)[3] // 2

        self.metaData['DataParameters'][0]['Dim'] = Dim

        headerPrint = yaml.dump(self.header.metaData)
        headerLength = len(headerPrint.encode('ascii'))

        with open(filePath, "wb") as exportFile:
            # exportFile = open(filePath, "wb")
            exportFile.write(f"{headerLength}".encode('ascii'))
            exportFile.write(headerPrint.encode('ascii'))
            # for arr in self.data:
            exportFile.write(self.data.tobytes())

            exportFile.close()

        # Update the file to use the updated file
        self.filePath = filePath
        self.header = OssiviewDataHeader(filePath)
        self.data = self._getDataV2()