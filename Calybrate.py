import cv2 #openCV
import exiftool
import os, glob
import numpy as np
import pyzbar.pyzbar as pyzbar
from skimage.transform import warp,matrix_transform,resize,FundamentalMatrixTransform,estimate_transform,ProjectiveTransform
import re
import os
import time
import json
import matplotlib as cm
from PIL import Image as pilImage
from IPython import get_ipython
from pathlib import Path
from IPython.display import display
import pandas as pd
from skimage.transform import ProjectiveTransform
import matplotlib.pyplot as plt
from ipywidgets import FloatProgress, Layout
from IPython.display import display
import math
import micasense.imageset as imageset
from micasense.image import Image
from micasense.panel import Panel
import micasense.imageutils as imageutils
import micasense.utils as msutils
import micasense.plotutils as plotutils
import micasense.metadata as metadata
import micasense.capture as capture
import skimage
from skimage.transform import warp,matrix_transform,resize,FundamentalMatrixTransform,estimate_transform,ProjectiveTransform




ipython = get_ipython()
plt.rcParams["figure.facecolor"] = "w"
panelNames = None
imagePath = Path("./data/ALTUM-PT")
figsize=(30,23) # use this size for full-image-resolution display
# figsize=(16,13)   # use this size for export-sized display

def testEnv():
    print()
    print("Testing the required libraries...")
    print()
    print("Successfully imported all required libraries.")

    if os.name == 'nt':
        if os.environ.get('exiftoolpath') is None:
            print("Set the `exiftoolpath` environment variable as described above")
        else:
            if not os.path.isfile(os.environ.get('exiftoolpath')):
                print("The provided exiftoolpath isn't a file, check the settings")

    try:
        with exiftool.ExifTool(os.environ.get('exiftoolpath')) as exift:
            print('Successfully executed exiftool.')
            print()
    except Exception as e:
        print("Exiftool isn't working. Double check that you've followed the instructions above.")
        print("The execption text below may help to find the source of the problem:")
        print()
        print(e)
        import ipdb; ipdb.set_trace()

def testPanelDetection():
    imageName = glob.glob(os.path.join(imagePath,'IMG_0000_1.tif'))[0]
    print('Testing Panel Detection...')
    print()
    img = Image(imageName)
    img.plot_raw(figsize=(8.73,8.73));
    panel = Panel(img)  
    if not panel.panel_detected():
        raise IOError("Panel Not Detected! Check your installation of pyzbar")
    else:
        print('Panel Succesfully Detected!')
        print()
        panel.plot(figsize=(8,8));

def testing():
    print("Welcome to the Calybrate Image Processing Module.")
    testEnv()
    testPanelDetection()
    print("Testing completed.\n")

def getPrefixes(image_paths):
     # Initialize a set to store unique prefixes
    prefixes = set()

    # Extract the prefixes from the image paths
    for path in image_paths:
        filename = os.path.basename(path)
        match = re.match(r'(IMG_\d{4})_\d\.tif', filename)
        if match:
            prefixes.add(match.group(1))

    # Sort the prefixes
    sorted_prefixes = sorted(prefixes)

    # Create a list of patterns based on the sorted prefixes
    image_patterns = [f"{prefix}_*.tif" for prefix in sorted_prefixes]
    return image_patterns

def multispectralBandHistogram(thecapture, im_aligned, panchroCam, sharpened_stack):
    theColors = {'Blue': 'blue', 'Green': 'green', 'Red': 'red', \
             'Red edge': 'maroon', 'NIR': 'purple', 'Panchro': 'yellow', 'PanchroB': 'orange',\
            'Red edge-740': 'salmon', 'Red Edge': 'maroon', 'Blue-444': 'aqua', \
             'Green-531': 'lime', 'Red-650': 'lightcoral', 'Red edge-705':'brown'}

    eo_count = len(thecapture.eo_indices())
    multispec_min = np.min(np.percentile(im_aligned[:,:,1:eo_count].flatten(),0.01))
    multispec_max = np.max(np.percentile(im_aligned[:,:,1:eo_count].flatten(), 99.99))

    theRange = (multispec_min,multispec_max)

    fig, axis = plt.subplots(1, 1, figsize=(10,4))
    for x,y in zip(thecapture.eo_indices(),thecapture.eo_band_names()):
        axis.hist(im_aligned[:,:,x].ravel(), bins=512, range=theRange, \
                histtype="step", label=y, color=theColors[y], linewidth=1.5)
    plt.title("Multispectral histogram (radiance)")
    axis.legend()
    plt.savefig('Results/Calibration/Multispectral_histogram.png')

    if panchroCam:
        eo_count = len(thecapture.eo_indices())
        multispec_min = np.min(np.percentile(sharpened_stack[:,:,1:eo_count].flatten(),0.01))
        multispec_max = np.max(np.percentile(sharpened_stack[:,:,1:eo_count].flatten(), 99.99))

        theRange = (multispec_min,multispec_max)

        fig, axis = plt.subplots(1, 1, figsize=(10,4))
        for x,y in zip(thecapture.eo_indices(),thecapture.eo_band_names()):
            axis.hist(sharpened_stack[:,:,x].ravel(), bins=512, range=theRange, \
                    histtype="step", label=y, color=theColors[y], linewidth=1.5)
        plt.title("Pan-sharpened multispectral histogram (radiance)")
        axis.legend()
        plt.savefig('Results/Calibration/pan_sharpened_Multispectral_histogram.png')

def saveVisualizationOfAlignetImages(thecapture, im_aligned, panchroCam, sharpened_stack):
    

    rgb_band_indices = [thecapture.band_names_lower().index('red'),
                        thecapture.band_names_lower().index('green'),
                        thecapture.band_names_lower().index('blue')]
    cir_band_indices = [thecapture.band_names_lower().index('nir'),
                        thecapture.band_names_lower().index('red'),
                        thecapture.band_names_lower().index('green')]

    # Create normalized stacks for viewing
    im_display = np.zeros((im_aligned.shape[0],im_aligned.shape[1],im_aligned.shape[2]), dtype=np.float32)
    im_min = np.percentile(im_aligned[:,:,rgb_band_indices].flatten(), 0.5)  # modify these percentiles to adjust contrast
    im_max = np.percentile(im_aligned[:,:,rgb_band_indices].flatten(), 99.5)  # for many images, 0.5 and 99.5 are good values

    if panchroCam:
        im_display_sharp = np.zeros((sharpened_stack.shape[0],sharpened_stack.shape[1],sharpened_stack.shape[2]), dtype=np.float32 )
        im_min_sharp = np.percentile(sharpened_stack[:,:,rgb_band_indices].flatten(), 0.5)  # modify these percentiles to adjust contrast
        im_max_sharp = np.percentile(sharpened_stack[:,:,rgb_band_indices].flatten(), 99.5)  # for many images, 0.5 and 99.5 are good values


    # for rgb true color, we use the same min and max scaling across the 3 bands to 
    # maintain the "white balance" of the calibrated image
    for i in rgb_band_indices:
        im_display[:,:,i] =  imageutils.normalize(im_aligned[:,:,i], im_min, im_max)
        if panchroCam: 
            im_display_sharp[:,:,i] = imageutils.normalize(sharpened_stack[:,:,i], im_min_sharp, im_max_sharp)

    rgb = im_display[:,:,rgb_band_indices]

    if panchroCam:
        rgb_sharp = im_display_sharp[:,:,rgb_band_indices]

    return rgb_sharp

def imageEnhancement(panchroCam, rgb_sharp, currImageName):
    if panchroCam:
        rgb = rgb_sharp
    # Create an enhanced version of the RGB render using an unsharp mask
    gaussian_rgb = cv2.GaussianBlur(rgb, (9,9), 10.0)
    gaussian_rgb[gaussian_rgb<0] = 0
    gaussian_rgb[gaussian_rgb>1] = 1
    unsharp_rgb = cv2.addWeighted(rgb, 1.5, gaussian_rgb, -0.5, 0)
    unsharp_rgb[unsharp_rgb<0] = 0
    unsharp_rgb[unsharp_rgb>1] = 1

    # Apply a gamma correction to make the render appear closer to what our eyes would see
    gamma = 1.4
    gamma_corr_rgb = unsharp_rgb**(1.0/gamma)
    fig = plt.figure(figsize=figsize)
    cv2.imwrite('Results/EnhancedImages/'+currImageName+'-enhanced.png', cv2.cvtColor(gamma_corr_rgb*255, cv2.COLOR_RGB2BGR))

    # plt.imshow(gamma_corr_rgb, aspect='equal')
    # plt.axis('off')
    # plt.savefig('Results/EnhancedImages/'+thecapture.uuid+'-enhanced2.png')
    return gamma_corr_rgb

def stackExport(thecapture, panchroCam, currImageName):
    #set output name to unique capture ID, e.g. FWoNSvgDNBX63Xv378qs
    outputName = currImageName

    st = time.time()
    if panchroCam:
        # in this example, we can export both a pan-sharpened stack and an upsampled stack
        # so you can compare them in GIS. In practice, you would typically only output the pansharpened stack 
        print(outputName+"-pansharpened.tif")
        thecapture.save_capture_as_stack("Results/Pan Sharpened Stacks/"+thecapture.uuid+".tif", sort_by_wavelength=True, pansharpen=True)
        thecapture.save_capture_as_stack("Results/Pan Sharpened Stacks/"+outputName+"-upsampled.tif", sort_by_wavelength=True, pansharpen=False)
    else:
        thecapture.save_capture_as_stack(outputName+"-noPanels.tif", sort_by_wavelength=True)

    et = time.time()
    elapsed_time = et - st
    print("Time to save stacks:", int(elapsed_time), "seconds.")

def save_mask_and_overlay(imgbase_path, imgcolor, save_mask_path, save_overlay_path, vmin, vmax,colormap='viridis'):
    # Load the base image from the saved PNG file
    imgbase = pilImage.open(imgbase_path).convert("RGB")
    imgbase = np.array(imgbase)

    # Create the mask: mask where imgcolor is NaN or outside the vmin, vmax range
    mask = np.ma.getmaskarray(np.ma.masked_invalid(imgcolor))
    mask = mask.astype(np.uint8) * 255  # 255 where mask is invalid (NaN or < vmin)

    # Save the mask
    mask_image = pilImage.fromarray(mask)
    mask_image.save(save_mask_path)

    # Normalize imgcolor to range [0, 1] based on vmin and vmax
    imgcolor_normalized = (imgcolor - vmin) / (vmax - vmin)
    imgcolor_normalized = np.clip(imgcolor_normalized, 0, 1)  # Clip values outside [0, 1]

    # Apply the colormap
    cmap = cm.colormaps[colormap]
    imgcolor_mapped = cmap(imgcolor_normalized)

    # Convert colormap image to uint8
    imgcolor_rgb = (imgcolor_mapped[:, :, :3] * 255).astype(np.uint8)  # Drop the alpha channel and scale to 255

    # Ensure imgbase is RGB (3 channels)
    imgbase_rgb = imgbase[:, :, :3] if imgbase.shape[2] == 4 else imgbase

    # Apply the colormap only where the mask is 0 (indicating valid data)
    overlay = np.where(mask[:, :, None] == 0, imgcolor_rgb, imgbase_rgb)

    # Save the overlay
    overlay_image = pilImage.fromarray(overlay.astype(np.uint8))
    overlay_image.save(save_overlay_path)

def imageAlignment(thecapture, irradiance_list,img_type, warp_matrices_SIFT, currImageName):
    print("Aligning image "+ currImageName + "...")

    st = time.time()

    sharpened_stack, upsampled = thecapture.radiometric_pan_sharpened_aligned_capture(warp_matrices=warp_matrices_SIFT, irradiance_list=irradiance_list, img_type=img_type)
        
    # we can also use the Rig Relatives from the image metadata to do a quick, rudimentary alignment 
    #     warp_matrices0=thecapture.get_warp_matrices(ref_index=5)
    #     sharpened_stack,upsampled = radiometric_pan_sharpen(thecapture,warp_matrices=warp_matrices0)
    
    print("Pansharpened shape:", sharpened_stack.shape)
    print("Upsampled shape:", upsampled.shape)
    # re-assign to im_aligned to match rest of code 
    im_aligned = upsampled

    stackExport(thecapture, True, thecapture.uuid)
    et = time.time()
    elapsed_time = et - st
    print('\nAlignment and pan-sharpening time:', int(elapsed_time), 'seconds\n')
    
    return im_aligned, sharpened_stack

def panelCalybration(panelImageName, currImageName):
    print("Calybration process started.\n")
    if '__IPYTHON__' in globals():
        ipython.magic('load_ext autoreload')
        ipython.magic('autoreload 2')
        ipython.magic('matplotlib inline')
    # these will return lists of image paths as strings 
    panelNames = list(imagePath.glob(panelImageName))
    panelNames = [x.as_posix() for x in panelNames]

    imageNames = list(imagePath.glob(currImageName))
    imageNames = [x.as_posix() for x in imageNames]

    if panelNames is not None:
        panelCap = capture.Capture.from_filelist(panelNames)
    else:
        panelCap = None

    thecapture = capture.Capture.from_filelist(imageNames)

    if len(thecapture.camera_serials) > 1:
        cam_serial = "_".join(thecapture.camera_serials)
        print(cam_serial)
    else:
        cam_serial = thecapture.camera_serial
        
    # print("Camera model:",cam_model)
    # print("Bit depth:", thecapture.bits_per_pixel)
    # print("Camera serial number:", cam_serial)
    # print("Capture ID:",thecapture.uuid)

    # get camera model for future use 
    cam_model = thecapture.camera_model

    # determine if this sensor has a panchromatic band 
    if cam_model == 'RedEdge-P' or cam_model == 'Altum-PT':
        panchroCam = True
    else:
        panchroCam = False
        panSharpen = False 

    if panelCap is not None:
        if panelCap.panel_albedo() is not None:
            panel_reflectance_by_band = panelCap.panel_albedo()
        else:
            panel_reflectance_by_band = [0.49]*len(thecapture.eo_band_names()) #RedEdge band_index order
        panel_irradiance = panelCap.panel_irradiance(panel_reflectance_by_band)  
        irradiance_list = panelCap.panel_irradiance(panel_reflectance_by_band) + [0] # add to account for uncalibrated LWIR band, if applicable
        img_type = "reflectance"
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
    if panchroCam:
        warp_matrices_filename = cam_serial + "_warp_matrices_SIFT.npy"
    else:
        warp_matrices_filename = cam_serial + "_warp_matrices_opencv.npy"

    if Path('./' + warp_matrices_filename).is_file():
        print("Found existing warp matrices for camera", cam_serial)
        load_warp_matrices = np.load(warp_matrices_filename, allow_pickle=True)
        loaded_warp_matrices = []
        for matrix in load_warp_matrices: 
            if panchroCam:
                transform = ProjectiveTransform(matrix=matrix.astype('float64'))
                loaded_warp_matrices.append(transform)
            else:
                loaded_warp_matrices.append(matrix.astype('float32'))
        print("Warp matrices successfully loaded.")

        if panchroCam:
            warp_matrices_SIFT = loaded_warp_matrices
        else:
            warp_matrices = loaded_warp_matrices
    else:
        print("No existing warp matrices found. Creating a new one...")
        warp_matrices_SIFT = False
        warp_matrices = False

    if panchroCam: 
        st = time.time()
        if not warp_matrices_SIFT :
            print("Generating new warp matrices...")
            warp_matrices_SIFT = thecapture.SIFT_align_capture(min_matches = 10)
        et = time.time()
        elapsed_time = et - st
        print('Alignment and pan-sharpening time:', int(elapsed_time), 'seconds')
    if panchroCam:
        working_wm = warp_matrices_SIFT
    else:
        working_wm = warp_matrices
    if not Path('./' + warp_matrices_filename).is_file():
        temp_matrices = []
        for x in working_wm:
            if isinstance(x, np.ndarray):
                temp_matrices.append(x)
            if isinstance(x, skimage.transform._geometric.ProjectiveTransform):
                temp_matrices.append(x.params)
        np.save(warp_matrices_filename, np.array(temp_matrices, dtype=object), allow_pickle=True)
        print("Saved to", Path('./' + warp_matrices_filename).resolve())
    else:
        print("Matrices already exist at",Path('./' + warp_matrices_filename).resolve())
            
    im_aligned, sharpened_stack = imageAlignment(thecapture, irradiance_list,img_type, warp_matrices_SIFT, "Calibration")

    multispectralBandHistogram(thecapture, im_aligned, panchroCam, sharpened_stack)

    return thecapture, panchroCam,img_type,irradiance_list,warp_matrices_SIFT

def NDVIComputation(thecapture, im_aligned, panchroCam, sharpened_stack, img_type, gamma_corr_rgb, imageName, band='red'):
    figsize=(30,23)

    if band=='red':
        calculateIndex='NDVI'
    elif band=='green':
        calculateIndex='GNDVI'
    elif band=='blue':
        calculateIndex='BNDVI'
    
    print("\nCalculating ",calculateIndex,"...")

    nir_band = thecapture.band_names_lower().index('nir')
    selected_band = thecapture.band_names_lower().index(band)

    thelayer = im_aligned
    if panchroCam:
        thelayer = sharpened_stack
    np.seterr(divide='ignore', invalid='ignore') # ignore divide by zero errors in the index calculation

    # Compute Normalized Difference Vegetation Index (NDVI) from the NIR(3) and RED (2) bands
    ndvi = (thelayer[:,:,nir_band] - thelayer[:,:,selected_band]) / (thelayer[:,:,nir_band] + thelayer[:,:,selected_band])
    # print("Image type:",img_type)

    # remove shadowed areas (mask pixels with NIR reflectance < 20%))
    # this does not seem to work on panchro stacks 
    if img_type == 'reflectance':
        ndvi = np.ma.masked_where(thelayer[:,:,nir_band] < 0.20, ndvi)
    elif img_type == 'radiance':
        lower_pct_radiance = np.percentile(thelayer[:,:,nir_band],  10.0)
        ndvi = np.ma.masked_where(thelayer[:,:,nir_band] < lower_pct_radiance, ndvi)
    
    ndviCopy = np.copy(ndvi)
    ndvi_hist_min = np.min(np.percentile(ndviCopy,0.5))
    ndvi_hist_max = np.max(np.percentile(ndviCopy,99.5))

    fig, axis = plt.subplots(1, 1, figsize=(10,4))

    axis.hist(ndvi.ravel(), bins=512, range=(ndvi_hist_min, ndvi_hist_max))
    
    
    
    plt.title(calculateIndex+" Histogram")
    baseFolder='Results/Indexes/'+calculateIndex+'/'
    plt.savefig(baseFolder+'Histogram/'+imageName+'_histogram.png')

    min_display_ndvi = 0.45 # further mask soil by removing low-ndvi values
    # min_display_ndvi = np.percentile(ndviCopy.flatten(),  5.0)  # modify with these percentilse to adjust contrast
    max_display_ndvi = np.percentile(ndviCopy.flatten(), 99.5)  # for many images, 0.5 and 99.5 are good values
    masked_ndvi = np.ma.masked_where(ndvi < min_display_ndvi, ndvi)

    # #reduce the figure size to account for colorbar
    figsize=np.asarray(figsize) - np.array([3,2])

    #plot NDVI over an RGB basemap, with a colorbar showing the NDVI scale

    print(min_display_ndvi, max_display_ndvi)
    fig, axis = plotutils.plot_overlay_withcolorbar(gamma_corr_rgb, 
                                        masked_ndvi, 
                                        figsize = (14,7), 
                                        title = calculateIndex + ' filtered to only plants over RGB base layer',
                                        vmin = min_display_ndvi,
                                        vmax = max_display_ndvi,
                                        show=False)
    
    fig.savefig(baseFolder+ "Fig/"+imageName+'_'+calculateIndex+'_over_rgb.png')
    save_mask_and_overlay('Results/EnhancedImages/'+imageName+'-enhanced.png',
                           masked_ndvi,
                           baseFolder+"Mask/"+ imageName+'_Mask.png',
                           baseFolder+'Overlay/'+imageName+'_Overlay.png',
                           min_display_ndvi,
                           max_display_ndvi)
    print(calculateIndex," Computation completed.\n")

    return ndvi

def NDREComputation(thecapture, im_aligned, gamma_corr_rgb, ndvi, imageName):
    # Compute Normalized Difference Red Edge Index from the NIR(3) and RedEdge(4) bands
    figsize=(30,23)
    nir_band = thecapture.band_names_lower().index('nir')
    rededge_band = thecapture.band_names_lower().index('red edge')

    thelayer = im_aligned

    min_display_ndvi = 0.45 # further mask soil by removing low-ndvi values

    ndre = (thelayer[:,:,nir_band] - thelayer[:,:,rededge_band]) / (thelayer[:,:,nir_band] + thelayer[:,:,rededge_band])

    # Mask areas with shadows and low NDVI to remove soil
    
    masked_ndre = np.ma.masked_where(ndvi < min_display_ndvi, ndre)

    maskedNdreCopy= np.copy(masked_ndre)

    # Compute a histogram
    ndre_hist_min = np.min(np.percentile(maskedNdreCopy,0.5))
    ndre_hist_max = np.max(np.percentile(maskedNdreCopy,99.5))

    fig, axis = plt.subplots(1, 1, figsize=(10,4))
    axis.hist(masked_ndre.ravel(), bins=512, range=(ndre_hist_min, ndre_hist_max))
    plt.title("NDRE Histogram (filtered to only plants)")
    plt.savefig('Results/Indexes/NDRE/Histogram/'+imageName+'_ndre_histogram.png')

    
    min_display_ndre = np.percentile(maskedNdreCopy, 5)
    max_display_ndre = np.percentile(maskedNdreCopy, 99.5)

    baseFolder='Results/Indexes/NDRE/'

    fig, axis = plotutils.plot_overlay_withcolorbar(gamma_corr_rgb, 
                                        masked_ndre, 
                                        figsize=(14,7), 
                                        title='NDRE filtered to only plants over RGB base layer',
                                        vmin=min_display_ndre,vmax=max_display_ndre,show=False)
    fig.savefig(baseFolder+"Fig/"+imageName+'_ndre_over_rgb.png')

    save_mask_and_overlay('Results/EnhancedImages/'+imageName+'-enhanced.png',
                           masked_ndre,
                           baseFolder+"Mask/"+ imageName+'_Mask.png',
                           baseFolder+'Overlay/'+imageName+'_Overlay.png',
                           min_display_ndre,
                           max_display_ndre)

    print("NDRE Computation completed.\n")

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

    data, columns = imgset.as_nested_lists()
    print("Columns: {}".format(columns))
    max_lat = max([point[1] for point in data])
    min_lat = min([point[1] for point in data])
    max_lon = max([point[2] for point in data])
    min_lon = min([point[2] for point in data])
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


def calculateIndexes(thecapture, panchroCam ,img_type, irradiance_list,warp_matrices_SIFT, saveName):
    im_aligned, sharpened_stack = imageAlignment(thecapture, irradiance_list,img_type, warp_matrices_SIFT, saveName)

    rgb=saveVisualizationOfAlignetImages(thecapture, im_aligned, panchroCam, sharpened_stack)
    gama_corr_rgb = imageEnhancement(panchroCam, rgb, saveName)
    # stackExport(thecapture, panchroCam, saveName)
    ndvi=NDVIComputation(thecapture, im_aligned , panchroCam, sharpened_stack,img_type,gama_corr_rgb, saveName, band='red')
    # NDVIComputation(thecapture, im_aligned , panchroCam, sharpened_stack,img_type,gama_corr_rgb, saveName, band='green')
    # NDVIComputation(thecapture, im_aligned , panchroCam, sharpened_stack,img_type,gama_corr_rgb, saveName, band='blue')
    # NDREComputation(thecapture, im_aligned, gama_corr_rgb, ndvi, saveName)
    plt.close('all')
    



def main(): 
    panelImageName = 'IMG_0000_*.tif'
    currImageName = 'IMG_0003_*.tif'

    all_images = glob.glob(os.path.join(imagePath, '*'))

    imageNames=getPrefixes(all_images)  
    imageNames.remove(panelImageName)

    # testing()
    # createGeoJson()

    thecapture, panchroCam,img_type,irradiance_list,warp_matrices_SIFT = panelCalybration(panelImageName, currImageName)

    st = time.time()
    for imageName in imageNames:
        saveName=imageName[:-6]
        currImageNames = list(imagePath.glob(imageName))
        currImageNames = [x.as_posix() for x in currImageNames]
        thecapture = capture.Capture.from_filelist(currImageNames)
        calculateIndexes(thecapture, panchroCam,img_type, irradiance_list, warp_matrices_SIFT, saveName)
    et = time.time()
    elapsed_time = et - st
    print("Total time for calculating indexes", int(elapsed_time), 'seconds\n')
    
if __name__=="__main__": 
    main() 