__author__ = 'Marc Bickel'

from peewee import *

import os
import sys

import struct
from natsort import natsorted

__created__ = '08.04.2015'
__updated__ = '24.06.2015'

db = SqliteDatabase(None)

def init_db(sqlite_path):
    '''
    Open SQLite database at given path und connect to it.
    '''

    if not os.path.isfile(sqlite_path):
        raise DoesNotExist("Error: Database not found at: " + sqlite_path)

    db.init(sqlite_path)
    db.connect()

    # check if all tables are available
    tables = db.get_tables()

    if 'Images' not in tables:
        db.create_table(Images)

    if 'Patients' not in tables:
        db.create_table(Patients)

    if 'patients_add' not in tables:
        db.create_table(Patients_Add)

    if 'Series' not in tables:
        db.create_table(Series)

    if 'series_add' not in tables:
        db.create_table(Series_Add)

    if 'Studies' not in tables:
        db.create_table(Studies)

    if 'niftis' not in tables:
        db.create_table(Niftis)

    if 'segmented' not in tables:
        db.create_table(Segmented)

def close_db():
    if db is not None:
        db.close()

class BaseModel(Model):
    class Meta:
        database = db

class Images(BaseModel):
    sopInstanceUID = CharField(primary_key=True, max_length=64)
    filename = CharField(max_length=1024)
    seriesInstanceUID = CharField(max_length=64)
    insertTimestamp = DateTimeField()

class Patients(BaseModel):
    # original 3D-Slicer columns
    uid = IntegerField(primary_key=True)
    patientsName = CharField(max_length=255)
    patientID = CharField(max_length=255)
    patientsBirthDate = DateTimeField()
    patientsBirthTime = DateTimeField()
    patientsSex = CharField(max_length=1)
    patientsAge = CharField(max_length=10)
    patientsComments = CharField(max_length=255)

class Patients_Add(BaseModel):
    uid = IntegerField(primary_key=True)

    # additional fields
    patientIdentityRemoved = CharField(max_length=64, null=True)
    patientPosition = CharField(max_length=64, null=True)
    patientSize = FloatField(null=True)
    patientWeight = IntegerField(null=True)

class Series(BaseModel):
    # original 3D-Slicer columns
    seriesInstanceUID = CharField(primary_key=True, max_length=64)
    studyInstanceUID = CharField(max_length=64)
    seriesNumber = IntegerField()
    seriesDate = DateTimeField()
    seriesTime = DateTimeField()
    seriesDescription = CharField(max_length=255)
    modality = CharField(max_length=20)
    bodyPartExamined = CharField(max_length=255)
    frameOfReferenceUID = CharField(max_length=64)
    acquisitionNumber = IntegerField()
    contrastAgent = CharField(max_length=255)
    scanningSequence = CharField(max_length=45)
    echoNumber = IntegerField()
    temporalPosition = IntegerField()

class Niftis(BaseModel):
    # contains references to nifti data
    uid = IntegerField(primary_key=True)
    seriesInstanceUID = CharField(max_length=64, null=False)

    niftiFilename = CharField(max_length=1024, null=True)
    niftiFilenameOrdered = CharField(max_length=1024, null=True)
    niftiFilesize = IntegerField(null=True)
    niftiSliceCount = IntegerField(null=True)

class Series_Add(BaseModel):
    seriesInstanceUID = CharField(primary_key=True, max_length=64)

    # additional fields
    # general data
    acquisitionTime = CharField(max_length=64, null=True)
    cTDIvol = DoubleField(null=True)

    imageComments = CharField(max_length=255, null=True)

    institutionName = CharField(max_length=255, null=True)
    institutionAddress = CharField(max_length=255, null=True)
    stationName = CharField(max_length=255, null=True)

    manufacturer = CharField(max_length=255, null=True)
    manufacturerModelName = CharField(max_length=255, null=True)
    deviceSerialNumber = CharField(max_length=64, null=True)

    softwareVersions = CharField(max_length=255, null=True)
    specificCharacterSet = CharField(max_length=64, null=True)
    spiralPitchFactor = FloatField(null=True)

    # image data
    columns = IntegerField(null=True)
    rows = IntegerField(null=True)
    exposure = IntegerField(null=True)
    exposureTime = IntegerField(null=True)
    imageType = CharField(max_length=255, null=True)

    pixelRepresentation = IntegerField(null=True)
    pixelSpacing = CharField(max_length=255, null=True)

    samplesPerPixel = IntegerField(null=True)
    seriesNumber = IntegerField(null=True)
    singleCollimationWidth = FloatField(null=True)
    sliceThickness = IntegerField(null=True)

    smallestImagePixelValue = IntegerField(null=True)
    largestImagePixelValue = IntegerField(null=True)

    # position and ct-setup
    dataCollectionDiameter = IntegerField(null=True)

    planarConfiguration = IntegerField(null=True)
    protocolName = CharField(max_length=64, null=True)
    reconstructionDiameter = IntegerField(null=True)
    rescaleIntercept = IntegerField(null=True)
    rescaleSlope = IntegerField(null=True)
    rescaleType = CharField(max_length=64, null=True)
    rotationDirection = CharField(max_length=64, null=True)

    totalCollimationWidth = FloatField(null=True)
    windowCenter = CharField(max_length=64, null=True)
    windowWidth = CharField(max_length=64, null=True)

    # contrast medium
    contrastBolusAgent = CharField(max_length=64, null=True)
    contrastBolusIngredientConcentration = IntegerField(null=True)
    contrastBolusStartTime = TimeField(null=True)
    contrastBolusStopTime = TimeField(null=True)
    contrastBolusTotalDose = FloatField(null=True)
    contrastBolusVolume = FloatField(null=True)
    contrastFlowDuration = FloatField(null=True)
    contrastFlowRate = FloatField(null=True)

    # overlay
    numberOfFramesInOverlay = IntegerField(null=True)
    overlayBitPosition = FloatField(null=True)
    overlayBitsAllocated = IntegerField(null=True)

    overlayColumns = IntegerField(null=True)
    overlayRows = IntegerField(null=True)

    overlayData = CharField(max_length=255, null=True)
    overlayDescription = CharField(max_length=255, null=True)
    overlayOrigin = CharField(max_length=255, null=True)
    overlayType = CharField(max_length=255, null=True)


class Studies(BaseModel):
    studyInstanceUID = CharField(primary_key=True, max_length=64)
    patientsUID = IntegerField()
    studyID = CharField(max_length=255)
    studyDate = DateTimeField()
    studyTime = CharField(max_length=20)
    accessionNumber = CharField(max_length=255)
    modalitiesInStudy = CharField(max_length=255)
    institutionName = CharField(max_length=255)
    referringPhysician = CharField(max_length=255)
    performingPhysiciansName = CharField(max_length=255)
    studyDescription = CharField(max_length=255)

class Segmented(BaseModel):
    uid = IntegerField(primary_key=True)

    # nifti uid to match to ct volume
    nifti_uid = IntegerField(null=False)
    tspFilename = CharField(max_length=1024, null=True)
    ctNiftiFilename = CharField(max_length=1024, null=True)
    segNiftiFilename = CharField(max_length=1024, null=True)
    segQualityRating = IntegerField(null=True)
    liverTumors = BooleanField(null=True)

def mapAddData(Dcm, seriesInstanceUID, patientsUID, verbose=False):
    # init Series_Add Obj
    try:
        series = Series_Add.create(seriesInstanceUID=seriesInstanceUID)
    except:
        series = Series_Add.get(Series_Add.seriesInstanceUID == seriesInstanceUID)

    # general data
    val = getattr(Dcm, 'AcquisitionTime', False)
    if val:
        series.acquisitionTime = val

        if verbose:
            print('AcquisitionTime ' + val)

    val = getattr(Dcm, 'CTDIvol', False)
    if val:
        try:
            series.cTDIvol = float(val)
        except:
            sys.stderr.write('Error at Series ' + series.seriesInstanceUID + '. Skip value CTDIvol\n')

        if verbose:
            print('CTDIvol ' + str(val))

    val = getattr(Dcm, 'ImageComments', False)
    if val:
        if getattr(Dcm, 'SpecificCharacterSet', False) and (('IR 100' in Dcm.SpecificCharacterSet) or (Dcm.SpecificCharacterSet == 'ISO-8859-1')):
            series.imageComments = val.decode(encoding='iso-8859-1', errors='replace')
        else:
            series.imageComments = val.decode(encoding='utf-8', errors='replace')

        if verbose:
            print('ImageComments ' + val)

    val = getattr(Dcm, 'InstitutionAddress', False)
    if val:
        if getattr(Dcm, 'SpecificCharacterSet', False) and (('IR 100' in Dcm.SpecificCharacterSet) or (Dcm.SpecificCharacterSet == 'ISO-8859-1')):
            series.institutionAddress = val.decode(encoding='iso-8859-1', errors='replace')
        else:
            series.institutionAddress = val.decode(encoding='utf-8', errors='replace')

        if verbose:
            print('InstitutionAddress ' + val)

    val = getattr(Dcm, 'InstitutionName', False)
    if val:
        if getattr(Dcm, 'SpecificCharacterSet', False) and (('IR 100' in Dcm.SpecificCharacterSet) or (Dcm.SpecificCharacterSet == 'ISO-8859-1')):
            series.institutionName = val.decode(encoding='iso-8859-1', errors='replace')
        else:
            series.institutionName = val.decode(encoding='utf-8', errors='replace')

        if verbose:
            print('InstitutionName ' + val)

    val = getattr(Dcm, 'StationName', False)
    if val:
        if getattr(Dcm, 'SpecificCharacterSet', False) and (('IR 100' in Dcm.SpecificCharacterSet) or (Dcm.SpecificCharacterSet == 'ISO-8859-1')):
            series.stationName = val.decode(encoding='iso-8859-1', errors='replace')
        else:
            series.stationName = val.decode(encoding='utf-8', errors='replace')

        if verbose:
            print('StationName ' + val)

    val = getattr(Dcm, 'Manufacturer', False)
    if val:
        series.manufacturer = val.decode(encoding='utf-8', errors='replace')

        if verbose:
            print('Manufacturer ' + val)

    val = getattr(Dcm, 'ManufacturerModelName', False)
    if val:
        series.manufacturerModelName = val.decode(encoding='utf-8', errors='replace')

        if verbose:
            print('ManufacturerModelName ' + val)

    val = getattr(Dcm, 'DeviceSerialNumber', False)
    if val:
        try:
            series.deviceSerialNumber = str(val).decode(encoding='utf-8', errors='replace')
        except:
            sys.stderr.write('Error at Series ' + series.seriesInstanceUID + '. Skip value DeviceSerialNumber\n')

        if verbose:
            print('DeviceSerialNumber ' + val)

    val = getattr(Dcm, 'SoftwareVersions', False)
    if val:
        series.softwareVersions = str(val).decode(encoding='utf-8', errors='replace')

        if verbose:
            print('SoftwareVersions ' + str(val))

    val = getattr(Dcm, 'SpecificCharacterSet', False)
    if val:
        series.specificCharacterSet = str(val).decode(encoding='utf-8', errors='replace')

        if verbose:
            print('SpecificCharacterSet ' + val)

    val = getattr(Dcm, 'SpiralPitchFactor', False)
    if val:
        try:
            series.spiralPitchFactor = float(val)
        except:
            sys.stderr.write('Error at Series ' + series.seriesInstanceUID + '. Skip value SpiralPitchFactor\n')

        if verbose:
            print('SpiralPitchFactor ' + str(val))

    # image data
    val = getattr(Dcm, 'Columns', False)
    if val:
        try:
            series.columns = int(val)
        except:
            sys.stderr.write('Error at Series ' + series.seriesInstanceUID + '. Skip value Columns\n')

        if verbose:
            print('Columns ' + str(val))

    val = getattr(Dcm, 'Rows', False)
    if val:
        try:
            series.rows = int(val)
        except:
            sys.stderr.write('Error at Series ' + series.seriesInstanceUID + '. Skip value Rows\n')

        if verbose:
            print('Rows ' + str(val))

    val = getattr(Dcm, 'Exposure', False)
    if val:
        try:
            series.exposure = int(val)
        except:
            sys.stderr.write('Error at Series ' + series.seriesInstanceUID + '. Skip value Exposure\n')

        if verbose:
            print('Exposure ' + str(val))

    val = getattr(Dcm, 'ExposureTime', False)
    if val:
        try:
            series.exposureTime = int(val)
        except:
            sys.stderr.write('Error at Series ' + series.seriesInstanceUID + '. Skip value ExposureTime\n')

        if verbose:
            print('ExposureTime ' + str(val))

    val = getattr(Dcm, 'ImageType', False)
    if val:
        series.imageType = str(val).decode(encoding='utf-8', errors='replace')

        if verbose:
            print('ImageType ' + str(val))

    val = getattr(Dcm, 'PixelRepresentation', False)
    if val:
        try:
            series.pixelRepresentation = int(val)
        except:
            sys.stderr.write('Error at Series ' + series.seriesInstanceUID + '. Skip value PixelRepresentation\n')

        if verbose:
            print('PixelRepresentation ' + str(val))

    val = getattr(Dcm, 'PixelSpacing', False)
    if val:
        series.pixelSpacing = str(val).decode(encoding='utf-8', errors='replace')

        if verbose:
            print('PixelSpacing ' + str(val))

    val = getattr(Dcm, 'SamplesPerPixel', False)
    if val:
        try:
            series.samplesPerPixel = int(val)
        except:
            sys.stderr.write('Error at Series ' + series.seriesInstanceUID + '. Skip value SamplesPerPixel\n')

        if verbose:
            print('SamplesPerPixel ' + str(val))

    val = getattr(Dcm, 'SeriesNumber', False)
    if val:
        try:
            series.seriesNumber = int(val)
        except:
            sys.stderr.write('Error at Series ' + series.seriesInstanceUID + '. Skip value')

        if verbose:
            print('SeriesNumber ' + str(val))

    val = getattr(Dcm, 'SingleCollimationWidth', False)
    if val:
        try:
            series.singleCollimationWidth = float(val)
        except:
            sys.stderr.write('Error at Series ' + series.seriesInstanceUID + '. Skip value SingleCollimationWidth\n')

        if verbose:
            print('SingleCollimationWidth ' + str(val))

    val = getattr(Dcm, 'SliceThickness', False)
    if val:
        try:
            series.sliceThickness = int(val)
        except:
            sys.stderr.write('Error at Series ' + series.seriesInstanceUID + '. Skip value SliceThickness\n')

        if verbose:
            print('SliceThickness ' + str(val))

    val = getattr(Dcm, 'SmallestImagePixelValue', False)
    if val:
        try:
            series.smallestImagePixelValue = int(val)
        except:
            try:
                series.smallestImagePixelValue = int(struct.unpack_from('H', Dcm.SmallestImagePixelValue)[0])
            except:
                sys.stderr.write('Error at Series ' + series.seriesInstanceUID + '. Skip value SmallestImagePixelValue\n')

        if verbose:
            print('SmallestImagePixelValue ' + str(Dcm.SmallestImagePixelValue))

    val = getattr(Dcm, 'LargestImagePixelValue', False)
    if val:
        try:
            series.largestImagePixelValue = int(val)
        except:
            try:
                series.largestImagePixelValue = int(struct.unpack_from('H', Dcm.LargestImagePixelValue)[0])
            except:
                sys.stderr.write('Error at Series ' + series.seriesInstanceUID + '. Skip value LargestImagePixelValue\n')

        if verbose:
            print('LargestImagePixelValue ' + str(Dcm.LargestImagePixelValue))

    # position and ct-setup data
    val = getattr(Dcm, 'DataCollectionDiameter', False)
    if val:
        try:
            series.dataCollectionDiameter = int(val)
        except:
            sys.stderr.write('Error at Series ' + series.seriesInstanceUID + '. Skip value DataCollectionDiameter\n')

        if verbose:
            print('DataCollectionDiameter ' + str(val))

    val = getattr(Dcm, 'PlanarConfiguration', False)
    if val:
        try:
            series.planarConfiguration = int(val)
        except:
            sys.stderr.write('Error at Series ' + series.seriesInstanceUID + '. Skip value PlanarConfiguration\n')

        if verbose:
            print('PlanarConfiguration ' + str(val))

    val = getattr(Dcm, 'ProtocolName', False)
    if val:
        series.protocolName = val.decode(encoding='utf-8', errors='replace')

        if verbose:
            print('ProtocolName ' + val)

    val = getattr(Dcm, 'ReconstructionDiameter', False)
    if val:
        try:
            series.reconstructionDiameter = int(val)
        except:
            sys.stderr.write('Error at Series ' + series.seriesInstanceUID + '. Skip value ReconstructionDiameter\n')

        if verbose:
            print('ReconstructionDiameter ' + str(val))

    val = getattr(Dcm, 'RescaleIntercept', False)
    if val:
        try:
            series.rescaleIntercept = int(val)
        except:
            sys.stderr.write('Error at Series ' + series.seriesInstanceUID + '. Skip value RescaleIntercept\n')

        if verbose:
            print('RescaleIntercept ' + str(val))

    val = getattr(Dcm, 'RescaleSlope', False)
    if val:
        try:
            series.rescaleSlope = int(val)
        except:
            sys.stderr.write('Error at Series ' + series.seriesInstanceUID + '. Skip value RescaleSlope\n')

        if verbose:
            print('RescaleSlope ' + str(val))

    val = getattr(Dcm, 'RescaleType', False)
    if val:
        series.rescaleType = val.decode(encoding='utf-8', errors='replace')

        if verbose:
            print('RescaleType ' + val)

    val = getattr(Dcm, 'RotationDirection', False)
    if val:
        series.rotationDirection = val.decode(encoding='utf-8', errors='replace')

        if verbose:
            print('RotationDirection ' + val)

    val = getattr(Dcm, 'TotalCollimationWidth', False)
    if val:
        try:
            series.totalCollimationWidth = float(val)
        except:
            sys.stderr.write('Error at Series ' + series.seriesInstanceUID + '. Skip value TotalCollimationWidth\n')

        if verbose:
            print('TotalCollimationWidth ' + str(val))

    val = getattr(Dcm, 'WindowCenter', False)
    if val:
        series.windowCenter = str(val).decode(encoding='utf-8', errors='replace')

        if verbose:
            print('WindowCenter ' + str(val))

    val = getattr(Dcm, 'WindowWidth', False)
    if val:
        series.windowWidth = str(val).decode(encoding='utf-8', errors='replace')

        if verbose:
            print('WindowWidth ' + str(val))


    # contrast medium
    val = getattr(Dcm, 'ContrastBolusAgent', False)
    if val:
        series.contrastBolusAgent = val.decode(encoding='utf-8', errors='replace')

        if verbose:
            print('ContrastBolusAgent ' + val)

    val = getattr(Dcm, 'ContrastBolusIngredientConcentration', False)
    if val:
        try:
            series.contrastBolusIngredientConcentration = int(val)
        except:
            sys.stderr.write('Error at Series ' + series.seriesInstanceUID + '. Skip value ContrastBolusIngredientConcentration\n')

        if verbose:
            print('ContrastBolusIngredientConcentration ' + str(val))

    val = getattr(Dcm, 'ContrastBolusStartTime', False)
    if val:
        series.contrastBolusStartTime = val

        if verbose:
            print('ContrastBolusStartTime ' + str(val))

    val = getattr(Dcm, 'ContrastBolusStopTime', False)
    if val:
        series.contrastBolusStopTime = val

        if verbose:
            print('ContrastBolusStopTime ' + str(val))

    val = getattr(Dcm, 'ContrastBolusTotalDose', False)
    if val:
        try:
            series.contrastBolusTotalDose = float(val)
        except:
            sys.stderr.write('Error at Series ' + series.seriesInstanceUID + '. Skip value ContrastBolusTotalDose\n')

        if verbose:
            print('ContrastBolusTotalDose ' + str(val))

    val = getattr(Dcm, 'ContrastBolusVolume', False)
    if val:
        try:
            series.contrastBolusVolume = float(val)
        except:
            sys.stderr.write('Error at Series ' + series.seriesInstanceUID + '. Skip value ContrastBolusVolume\n')

        if verbose:
            print('ContrastBolusVolume ' + str(val))

    val = getattr(Dcm, 'ContrastFlowDuration', False)
    if val:
        try:
            series.contrastFlowDuration = float(val)
        except:
            sys.stderr.write('Error at Series ' + series.seriesInstanceUID + '. Skip value ContrastFlowDuration\n')

        if verbose:
            print('ContrastFlowDuration ' + str(val))

    val = getattr(Dcm, 'ContrastFlowRate', False)
    if val:
        try:
            series.contrastFlowRate = float(val)
        except:
            sys.stderr.write('Error at Series ' + series.seriesInstanceUID + '. Skip value ContrastFlowRate\n')

        if verbose:
            print('ContrastFlowRate ' + str(val))

    # overlay info
    val = getattr(Dcm, 'NumberOfFramesInOverlay', False)
    if val:
        try:
            series.numberOfFramesInOverlay = int(val)
        except:
            sys.stderr.write('Error at Series ' + series.seriesInstanceUID + '. Skip value NumberOfFramesInOverlay\n')

        if verbose:
            print('NumberOfFramesInOverlay ' + str(val))

    val = getattr(Dcm, 'OverlayBitPosition', False)
    if val:
        try:
            series.overlayBitPosition = float(val)
        except:
            sys.stderr.write('Error at Series ' + series.seriesInstanceUID + '. Skip value OverlayBitPosition\n')

        if verbose:
            print('OverlayBitPosition ' + str(val))

    val = getattr(Dcm, 'OverlayBitsAllocated', False)
    if val:
        try:
            series.overlayBitsAllocated = int(val)
        except:
            sys.stderr.write('Error at Series ' + series.seriesInstanceUID + '. Skip value OverlayBitsAllocated\n')

        if verbose:
            print('OverlayBitsAllocated ' + str(val))

    val = getattr(Dcm, 'OverlayColumns', False)
    if val:
        try:
            series.overlayColumns = int(val)
        except:
            sys.stderr.write('Error at Series ' + series.seriesInstanceUID + '. Skip value OverlayColumns\n')

        if verbose:
            print('OverlayColumns ' + str(val))

    val = getattr(Dcm, 'OverlayData', False)
    if val:
        series.overlayData = str(val).decode(encoding='utf-8', errors='replace')

        if verbose:
            print('OverlayData ' + val)

    val = getattr(Dcm, 'OverlayDescription', False)
    if val:
        series.overlayDescription = val.decode(encoding='utf-8', errors='replace')

        if verbose:
            print('OverlayDescription ' + val)

    val = getattr(Dcm, 'OverlayOrigin', False)
    if val:
        series.overlayOrigin = val.decode(encoding='utf-8', errors='replace')

        if verbose:
            print('OverlayOrigin ' + val)

    val = getattr(Dcm, 'OverlayRows', False)
    if val:
        try:
            series.overlayRows = int(val)
        except:
            sys.stderr.write('Error at Series ' + series.seriesInstanceUID + '. Skip value OverlayRows\n')

        if verbose:
            print('OverlayRows ' + str(val))

    val = getattr(Dcm, 'OverlayType', False)
    if val:
        series.overlayType = val.decode(encoding='utf-8', errors='replace')

        if verbose:
            print('OverlayType ' + val)

    # write to DB
    try:
        if series.is_dirty():
            series.save()
    except:
        sys.stderr.write('Error saving series: ' + str(series.seriesInstanceUID) + '\n')


    # init Patients_Add Obj
    try:
        pat = Patients_Add.get(Patients_Add.uid == patientsUID)
    except:
        pat = Patients_Add.create(uid=patientsUID)

    # patient data
    val = getattr(Dcm, 'PatientIdentityRemoved', False)
    if val:
        pat.patientIdentityRemoved = val.decode(encoding='utf-8', errors='replace')

        if verbose:
            print('PatientIdentityRemoved ' + val)

    val = getattr(Dcm, 'PatientPosition', False)
    if val:
        pat.patientPosition = val.decode(encoding='utf-8', errors='replace')

        if verbose:
            print('PatientPosition ' + val)

    val = getattr(Dcm, 'PatientSize', False)
    if val:
        try:
            pat.patientSize = float(val)
        except:
            sys.stderr.write('Error at Series ' + series.seriesInstanceUID + '. Skip value PatientSize\n')

        if verbose:
            print('PatientSize ' + str(val))

    val = getattr(Dcm, 'PatientWeight', False)
    if val:
        try:
            pat.patientWeight = int(val)
        except:
            sys.stderr.write('Error at Series ' + series.seriesInstanceUID + '. Skip value PatientWeight\n')

        if verbose:
            print('PatientWeight ' + str(val))

    # write to DB
    try:
        if pat.is_dirty():
            pat.save()
    except:
        sys.stderr.write('No data or error saving patient: ' + str(pat.uid) + '\n')


def getSeriesDicomPath(series):
    fnames = []
    for img in Images.select().where(Images.seriesInstanceUID == series.seriesInstanceUID):
        fnames.append(img.filename)


    if(len(fnames) > 0):
        return natsorted(fnames)[0]
    else:
        return False


def getPatientFromSeries(series):
    return Patients.get(Studies.get(series.studyInstanceUID == Studies.studyInstanceUID).patientsUID == Patients.uid)

