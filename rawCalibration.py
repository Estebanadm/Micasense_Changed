import cv2
import glob
import os
import re
import json
import time
import shutil

from pathlib import Path
from micasense.panel import Panel
from micasense.image import Image
from IPython import get_ipython
from skimage.transform import ProjectiveTransform

import numpy as np
import tifffile as tiff
from natsort import natsorted
import matplotlib.pyplot as plt
import micasense.imageset as imageset
import micasense.capture as capture
import micasense.metadata as metadata
import micasense.image as image
import micasense.utils as msutils
import micasense.plotutils as plotutils
from ipywidgets import FloatProgress, Layout



ipython = get_ipython()
imagePath = Path("/media/estebanduran/Big_Downloads/Micasense/0019SET/000")
panelCalibrationVal = { 
    "Blue": 0.49, 
    "Green": 0.49, 
    "Red": 0.49, 
    "Red edge": 0.49, 
    "NIR": 0.49,
    "Panchro": 0.49,
}


def createGeoJson():
    ## This progress widget is used for display of the long-running process
    print("Creating GeoJson file...")
    st = time.time()
    f = FloatProgress(min=0, max=1, layout=Layout(width='100%'), description="Loading")
    def update_f(val):
        if (val - f.value) > 0.005 or val == 1: #reduces cpu usage from updating the progressbar by 10x
            f.value=val

    images_dir = os.path.expanduser(os.path.join('.','data','ALTUM-PT')) 
    imgset = imageset.ImageSet.from_directory(images_dir, progress_callback=update_f)

    cameralat=0.00022252834935
    cameralon=0.00001661204636
    
    data, columns = imgset.as_nested_lists()
    print("Columns: {}".format(columns))
    max_lat = max([point[1] for point in data])+cameralon
    min_lat = min([point[1] for point in data])-cameralon
    max_lon = max([point[2] for point in data])+cameralat
    min_lon = min([point[2] for point in data])-cameralat
    print("Latitude range: {} to {}".format(min_lat, max_lat))
    print("Longitude range: {} to {}".format(min_lon, max_lon))

    # Define the GeoJSON structure in the specified format
    geojson = {
        "geodesic": False,
        "type": "Polygon",
        "coordinates": [
            [
                [min_lon, min_lat],  # Bottom-left corner
                [max_lon, min_lat],  # Bottom-right corner
                [max_lon, max_lat],  # Top-right corner
                [min_lon, max_lat],  # Top-left corner
                [min_lon, min_lat]   # Closing the polygon (same as bottom-left corner)
            ]
        ]
    }
    
    # Save the GeoJSON structure to a file
    output_file = 'Results/bounding_box.geojson'
    with open(output_file, 'w') as f:
        json.dump(geojson, f, indent=4)
 
    et = time.time()
    elapsed_time = et - st


    print(f"GeoJSON file '{output_file}' has been created in", int(elapsed_time), 'seconds\n')

def getMetadata(panelImageName):
    exiftoolPath = None
    if os.name == 'nt':
        exiftoolPath = os.environ.get('exiftoolpath')   
    # get image metadata
    meta = metadata.Metadata(panelImageName, exiftool_path=exiftoolPath)
    print(meta.get_all())
    cameraMake = meta.get_item('EXIF:Make')
    cameraModel = meta.get_item('EXIF:Model')
    firmwareVersion = meta.get_item('EXIF:Software')
    bandName = meta.get_item('XMP:BandName')
    # print(meta.get_all())
    # print('{0} {1} firmware version: {2}'.format(cameraMake, 
    #                                             cameraModel, 
    #                                             firmwareVersion))
    # print('Exposure Time: {0} seconds'.format(meta.get_item('EXIF:ExposureTime')))
    # # print('Imager Gain: {0}'.format(meta.get_item('EXIF:ISOSpeed')/100.0))
    # print('Size: {0}x{1} pixels'.format(meta.get_item('EXIF:ImageWidth'),meta.get_item('EXIF:ImageHeight')))
    # print('Band Name: {0}'.format(bandName))
    # print('Center Wavelength: {0} nm'.format(meta.get_item('XMP:CentralWavelength')))
    # print('Bandwidth: {0} nm'.format(meta.get_item('XMP:WavelengthFWHM')))
    # print('Capture ID: {0}'.format(meta.get_item('XMP:CaptureId')))
    # print('Flight ID: {0}'.format(meta.get_item('XMP:FlightId')))
    # print('Focal Length: {0}'.format(meta.get_item('XMP:FocalLength')))
    return meta

def panelCalibration(panelImageName,currImageName):
    # these will return lists of image paths as strings 
    imageNames = list(imagePath.glob(currImageName))
    imageNames = [x.as_posix() for x in imageNames]

    panelNames = list(imagePath.glob(panelImageName))
    panelNames = [x.as_posix() for x in panelNames]


    if panelNames is not None:
        panelCap = capture.Capture.from_filelist(panelNames)
    else:
        panelCap = None

    thecapture = capture.Capture.from_filelist(imageNames)

    # get camera model for future use 
    cam_model = thecapture.camera_model
    # if this is a multicamera system like the RedEdge-MX Dual,
    # we can combine the two serial numbers to help identify 
    # this camera system later. 
    if len(thecapture.camera_serials) > 1:
        cam_serial = "_".join(thecapture.camera_serials)
        print(cam_serial)
    else:
        cam_serial = thecapture.camera_serial
        
    print("Camera model:",cam_model)
    print("Bit depth:", thecapture.bits_per_pixel)
    print("Camera serial number:", cam_serial)
    print("Capture ID:",thecapture.uuid)


    if panelCap is not None:
        if panelCap.panel_albedo() is not None:
            panel_reflectance_by_band = panelCap.panel_albedo()
        else:
            panel_reflectance_by_bmagePathand = [0.49]*len(thecapture.eo_band_names()) #RedEdge band_index order
        print(thecapture.eo_band_names())
        panel_irradiance = panelCap.panel_irradiance(panel_reflectance_by_band)
        # thecapture.plot_undistorted_reflectance(panel_irradiance)
    else:
        if thecapture.dls_present():
            img_type='reflectance'
            irradiance_list = thecapture.dls_irradiance() + [0]
            thecapture.plot_undistorted_reflectance(thecapture.dls_irradiance())
        else:
            img_type = "radiance"
            thecapture.plot_undistorted_radiance() 
            irradiance_list = None
    print("Panel Irradiance:",panel_irradiance)
    return panel_irradiance  
        
def correctReflectance(currImageName, panel_irradiance):
    currImagePath = "./"+currImageName
    print(currImagePath)
    print(panel_irradiance)

    meta = getMetadata(currImagePath)
    index=  {  
        "Blue": 0, 
        "Green": 1, 
        "Red": 2, 
        "NIR": 3,
        "Red edge": 4, 
        "Panchro": 5,
    }
    bandName = meta.get_item('XMP:BandName')

    imageName=currImageName[14:]
    destinationFolder= '/home/estebanduran/Documents/GitHub/imageprocessing_Micasense/Results/Reflectance/'+imageName
    print(destinationFolder)
    
    if not bandName== 'LWIR':
        print('Band Name: {0}'.format(bandName), panel_irradiance[index[bandName]])
        thecapture = image.Image(currImageName)
        print(panel_irradiance)
        print(bandName)
        print(panel_irradiance[index[bandName]])
        correctedReflectance=thecapture.reflectance(irradiance=panel_irradiance[index[bandName]])
        tiff.imwrite(destinationFolder, correctedReflectance.astype(np.float32))
        thecapture.clear_image_data()
    else:
        shutil.copy(currImagePath, destinationFolder)

        # # plotutils.plotwithcolorbar(flightImageRaw, 'Raw Image')
        # radianceToReflectance = panelCalibrationVal[bandName] / panel_irradiance[index[bandName]]

        

        # flightRadianceImage, _, _, _ = msutils.raw_image_to_radiance(meta, flightImageRaw)
        # flightReflectanceImage = flightRadianceImage * radianceToReflectance
        # flightUndistortedReflectance = msutils.correct_lens_distortion(meta, flightReflectanceImage)
        # print(flightUndistortedReflectance)
        # cv2.imwrite(destinationFolder, flightUndistortedReflectance)
        # getMetadata(destinationFolder)
        # plotutils.plotwithcolorbar(flightUndistortedReflectance, 'Reflectance converted and undistorted image');

def checkMetadata():
    destinationFolder= '/home/estebanduran/Documents/GitHub/imageprocessing_Micasense/data/test'
    # destinationFolder= '/home/estebanduran/Documents/GitHub/imageprocessing_Micasense/data/ALTUM-PT'
    all_images = glob.glob(os.path.join(destinationFolder, '*'))
    for imageName in all_images:
        print(imageName)
        getMetadata(imageName)




def main(): 
    checkMetadata()
    panelImageName = 'IMG_0000_*.tif'
    currImageName = 'IMG_0003_*.tif'

    all_images = glob.glob(os.path.join(imagePath, '*'))
    
    filtered_images = [img for img in all_images if not os.path.basename(img).startswith('IMG_0000_')]

    filtered_images = natsorted(filtered_images)
 

    testing()
    createGeoJson()

    panel_irradiance = panelCalibration(panelImageName,currImageName)

    st = time.time()
    for imageName in filtered_images:
        correctReflectance(imageName, panel_irradiance)
        # import pdb; pdb.set_trace()

    et = time.time()
    elapsed_time = et - st
    print("Total time for calculating indexes", int(elapsed_time), 'seconds\n')
    

if __name__=="__main__": 
    main() 

'''
After saving all images run the folowing command to copy metadata from raw files to corrected files
/home/estebanduran/Documents/GitHub/imageprocessing_Micasense/Image-ExifTool-12.89/exiftool -config /home/estebanduran/Documents/GitHub/imageprocessing_Micasense/Image-ExifTool-12.89/config_files/MicaSense.config -TagsFromFile /home/estebanduran/Documents/GitHub/imageprocessing_Micasense/data/ALTUM-PT/%F -EXIF:All -XMP /home/estebanduran/Documents/GitHub/imageprocessing_Micasense/Results/Reflectance
'''