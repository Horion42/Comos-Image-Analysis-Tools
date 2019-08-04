import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import datetime
import time

from astropy import units as u
from astropy.io import ascii, fits 
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.stats import SigmaClip
from astropy.nddata import NDData, Cutout2D

from photutils import Background2D, SExtractorBackground, StdBackgroundRMS
from photutils import make_source_mask
from photutils import CircularAperture, SkyCircularAperture, aperture_photometry
from photutils.utils import calc_total_error

import zscale




def opens(imageFolder,targetlist,filters,
          frame='fk5',
          coord=(1,2)          
          ):
    
    """   
    Create the final table and sort the original images in the right order 
    
    Parameters
    ----------
    imageFolder: str
        Folder where are your images saved.
    targetlist: str
        Name of the ASCII file where the list of input objects is stored.
        First column is assumed to be the ID. Sky or image coordinates
        must be included. The table can contain other columns, 
        which will be ignored here.
    filters: list
        Name of filters.
    coord: tuple int (default: (1,2))
        Poistion of the `targetlist` columns (starting from 0) where coordinates
        are stored. 
        
    Return
    ------
    List of images in the right order.
    Table with id in first colunm and skycoord in second colunm.    
    """

    
    # load targets
    tbl0 = ascii.read(targetlist,comment="#",format='no_header')
    cnm = tbl0.colnames
    tbl0.rename_column(cnm[0],'id')
    tbl0.rename_column(cnm[coord[0]],'ra')
    tbl0.rename_column(cnm[coord[1]],'dec')
    tbl0['skycoord'] = SkyCoord(tbl0['ra'].tolist(),tbl0['dec'].tolist(),unit='deg', frame=frame) 
    
    # Create the final table
    tbl=tbl0['id','skycoord']
        
    # collect images
    fileMosaique=os.listdir(imageFolder)

    # copy to have the same size compared to the number of images
    image=fileMosaique.copy()    
    
    # Classify all images in the ordre of parametre filters
    for i in range(np.size(fileMosaique)): 
        for j in range(np.size(fileMosaique)):           
            if filters[i] in fileMosaique[j]:               
                image[i]=fileMosaique[j]
    
    return image, tbl






def flux(nddata, tbl, zeroPoint, gain, radius):
    """
    Derive the flux and the magnitude within circular aperture
    
    Parameters
    ----------
    nddata: numpy array
        Numpy array where is saved the data and the sky coordinates wcs.
    tbl: table
        Table where the first column is the id, the second the ra coordinate, the third
        is dec coordinate and the four the skycoordinate. 
    zeroPoint: float
        Zero point of your image
    gain: float
        The gain of your image
    radius: float 
        The radius of your circle in arcsec to derive the flux
    Return
    ------
    Table with id, skycoord, flux, flux error, magnitude, magnitude error       
    """
    
    # By convention, sextractor
    if gain==0.:
        
        gain=1000000000000000000000000000.
        
    
    result=tbl['id','skycoord']
    result['flux']=float(np.size(tbl))
    result['flux_err']=float(np.size(tbl))
    
    for i in range(np.size(tbl)):
        
        # Recover the position for each object 
        position=tbl['skycoord'][i]
        
        
        if hdr['NAXIS1'] >=50 and hdr['NAXIS2'] >=50: 
            
            # size of the background map            
            sizeBkg=50            
            # cut the mosaic in stamp to the wcs coordinate of your objects 
            cutout = Cutout2D(nddata.data, position, (sizeBkg, sizeBkg), wcs=nddata.wcs) 

            # Recover new data and wcs of the stamp            
            data=cutout.data
            wcs=cutout.wcs
            
            
        else:

            # size of the background map                        
            sizeBkg=min(hdr['NAXIS1'],hdr['NAXIS2'])
            
            # Keep data and wcs of the initial image
            data=nddata.data
            wcs=nddata.wcs

                        
        #########################
        ####### Background ######
        #########################
        
        # Mask sources
        mask = make_source_mask(data, snr=1, npixels=3, dilate_size=3)
        
        # Derive the background and the rms image
        bkg = Background2D(data,int(sizeBkg/10),filter_size=1,sigma_clip=None,
                               bkg_estimator=SExtractorBackground(SigmaClip(sigma=2.5)),bkgrms_estimator=StdBackgroundRMS(SigmaClip(sigma=2.5)),exclude_percentile=60, mask=mask)
        
        
        ###########################
        ###### Aperture Flux ######
        ###########################

        nddataStamp=NDData(data=data-bkg.background,wcs=wcs)        
      
        # Calculate the total error
        error=calc_total_error(cutout.data, bkg.background_rms, gain)
        
        # Define a cicularAperture in the wcs position of your objects        
        apertures = SkyCircularAperture(position, r=radius*u.arcsec)
    
        # Derive the flux and error flux
        phot_table = aperture_photometry(nddataStamp, apertures, error=error)
        phot_table['aperture_sum'].info.format = '%.8g'
        phot_table['aperture_sum_err'].info.format = '%.8g'
        
        # Recover data
        result['flux'][i]=phot_table['aperture_sum'][0]
        result['flux_err'][i]=phot_table['aperture_sum_err'][0]
        
        
        ###########################
        ######## Magnitude ########
        ###########################
        
    
    # convert flux into magnitude    
    result['mag']=-2.5*np.log10(result['flux']) + zeroPoint
    
    # convert flux error into magnitude error
    result['mag_err']=1.0857*(result['flux_err']/result['flux']) 
        
        
    
    return result







def stampFits(cutout, tbl, filters):
    
    """
    Print out a postage stamp at the list of objects in a folder: Stamp_fits/
    
    Parameters
    ----------
    cutout: list
        array of images N filt x N sources
    tbl: table
        Table where the first column is the id, the second the ra coordinate, the third
        is dec coordinate and the four the skycoordinate. 
    filters: list
        name of filters for the different image
    """
    
    # Look if a folder Stamp_fits/ exists (to save the stamps in .fits)
    dirpath1 = dirpath+'Stamp_fits/'
    if not os.path.exists(dirpath1): #to be super-cautious and prevent overwriting
        os.makedirs(dirpath1)
    
    now = datetime.datetime.now()  
    ampm = ('AM','PM')
    
    # Number for associate the good filters with the good stamp
    o=0 
     
    # Loop over each filter
    for i in cutout:
        
        print('Begin stamp'+filters[o])

        
        # Number for associate the good id with the good stamp        
        m=0
        # Loop over each source
        for stamp in i:
            
            
            # Recover wcs data and write header
            w=stamp.wcs
            header = w.to_header()
            
            # write fits file and the header
            hdus = fits.PrimaryHDU(stamp.data,header)             
            hdus.header['DATE_CUT'] = '{:d}-{:d}-{:d} {:02d}{}:{:02d}'.format(now.year,now.month,now.day,now.hour%12,ampm[now.hour//12],now.minute)
            hdus.header['OBJ_ID'] = str(tbl['id'][m])
            hdus.writeto(dirpath1+'{}-{:07d}.fits'.format(filters[o],tbl['id'][m]),overwrite=True)
            m=m+1
        o=o+1       
    return



def stack(cutout, radiusStack, gain, zeroPoint, wavelenght):
    """
    Print out pdf with stamp stacking and plot graph magnitude 
    in function of wavelenght
    
    Parameters
    ---------_
    cutout: list
        array of images N filt x N sources
    radiusStack: float
        radius in pixel
    zeroPoint: float
        Zero point of your image
    gain: float
        The gain of your image
    wavelenght: float
        wavelenght of each filters    
    """
    
    # define the resulting stacked image as a cutout with the same dimension of the input cutouts    
    o=0
    
    # Create pdf where are storing stack stamp and graph  
    pdfOut=PdfPages(dirpath+'stacking.pdf')
    
    fig, axs=plt.subplots(1,len(filters),figsize=(5,5))
    
    # Saving magnitude and error 
    mag=[[ 0.0 for i in range(len(filters))]for j in range(2)]

    for i in cutout:  # i is a list of galaxies
        
        print('Photometry '+filters[o])
        
        # Assuming that the shape is the same for all galaxies 
        #this is now a tuple, e.g. (25,25), so the code will work also for rectangular stamps
        sizeStack= i[0].shape 
        
        #this is now a tuple, e.g. (25,25), so the code will work also for rectangular stamps
        print(sizeStack)  #just for testing
        stackImg = np.zeros(sizeStack)   
        
        #LOOP over pixels: (I changed a lot here)
        # fait tous les pixel x
        for x in range(sizeStack[0]):
             # fait tous les pixel y
             for y in range(sizeStack[1]):

                 # for the given pixels, collect the flux from all stamps into a list
                 pxl = []  #use an empty list, but it can be also np.zeros()
                 for stamp in i:
                          pxl.append(stamp.data[x,y])
                 # caluclate the median:
                 stackImg[x,y] = np.median(pxl)        


        axs[o].set_title(filters[o],fontsize=5,pad=2.5)
        axs[o].get_xaxis().set_visible(False); axs[o].get_yaxis().set_visible(False)        
        mappa=axs[o].imshow(stackImg,cmap='afmhot',origin='lower',interpolation='nearest')        
        zrange=zscale.zscale(stackImg)  
        mappa.set_clim(zrange)

        
        
        
        
        # Mask sources
        mask = make_source_mask(stackImg, snr=1, npixels=3, dilate_size=3)
        
        # Derive the background and the rms image
        bkg = Background2D(stackImg,int(sizeStack[0]/10),filter_size=1,sigma_clip=None,
                               bkg_estimator=SExtractorBackground(SigmaClip(sigma=2.5)),bkgrms_estimator=StdBackgroundRMS(SigmaClip(sigma=2.5)),exclude_percentile=90, mask=mask)

        if gain[o]==0.:
        
            gain[o]=1000000000000000000000000000.



        # Calculate the total error
        error=calc_total_error(stackImg, bkg.background_rms, gain[o])
                
        # Define a cicularAperture in the wcs position of your objects        
        apertures = CircularAperture((int(sizeStack[0]/2),int(sizeStack[1]/2)), r=radiusStack[o])
        
        # Derive the flux and error flux
        photometry = aperture_photometry(stackImg-bkg.background, apertures, error=error)

        # Saving magnitude and error        
        mag[0][o]=-2.5*np.log10(photometry['aperture_sum'][0]) + zeroPoint[o]
        mag[1][o]=1.0857*(photometry['aperture_sum_err'][0]/photometry['aperture_sum'][0])

        o=o+1
    
    plt.savefig(pdfOut,format='pdf') 
    
    
    # Plot 
    plt.clf()
    
    y=mag[0]
    x=wavelenght
    plt.plot(x,y,'ro')
    
    xlimL=wavelenght[0]-1000
    xlimR=wavelenght[len(wavelenght)-1]+1000
    
    plt.xlim(xlimL,xlimR)
    plt.ylim(32,22)
    plt.errorbar(wavelenght,mag[0],yerr=mag[1],fmt='none', capsize = 10, ecolor = 'blue', zorder = 1)
    
    plt.xlabel('wavelenght(Å)')
    plt.ylabel('magnitude')
    plt.title('')
    
    plt.savefig(pdfOut,format='pdf')  
    pdfOut.close()
    np.savetxt(dirpath+'fluxStack.txt',mag,header='Ligne1: mag, Ligne2: mag_err')
    
    return







def PDF(cutout):
    
    """
    Create a pdf file with the plot of your objects in different filters
    
    Parameters
    ----------
    cutout: list
          array of images N filt x N sources
    """
            
    # create the pdf
    pdfOut=PdfPages(dirpath+'stampPDF.pdf')
    
    # it's a sum to derive the good number of stamp    
    p=0 
    
    # Placement of the Id
    sizeId=cutout[0][0].shape
    # 
    while p < np.size(cutout[0])-1:
        
        print(p)
    
        if np.size(cutout[0])-p > 10:                        
            fig, axs=plt.subplots(10,len(cutout),figsize=(8.3,11.7))
        else:
            fig, axs=plt.subplots(np.size(cutout[0])-p,len(cutout),figsize=(8.3,11.7))
             
        # Loop over the 10 sources to be include in one PDF page           
        for k in range(10):
            
            # id of object
            axs[k,0].text(-sizeId[0],sizeId[0]/2,tbl['id'][p])
            
            # Loop over the filters
            for j in range(len(cutout)):
                
                # Display the image
                mappa=axs[k,j].imshow(cutout[j][p].data,cmap='afmhot',origin='lower',interpolation='nearest')
                axs[k,j].set_title(filters[j],fontsize=5,pad=2.5)
                axs[k,j].get_xaxis().set_visible(False); axs[k,j].get_yaxis().set_visible(False)

                # Plot circular aperture at the coordinate of your object                                    
                apertures = SkyCircularAperture(tbl['skycoord'][p], r=1.5*u.arcsec)
                aperturesPix= apertures.to_pixel(cutout[j][p].wcs)
                aperturesPix.plot(color='cyan',ax=axs[k,j], lw=1, alpha=0.5)  

                # DS9 zscale
                zrange=zscale.zscale(cutout[j][p].data)  
                mappa.set_clim(zrange)

            # it's a sum to derive the good number of stamp
            if p < np.size(cutout[0])-1:
                p=p+1 
            else:                
                break
        
        # save the page 
        plt.savefig(pdfOut,format='pdf')  
    
    pdfOut.close()
    return




#################################################################################
#################################              ##################################
#################################   MAIN CODE  ##################################
#################################              ##################################
#################################################################################


    #############################################################
    ######################## PARAMETERS #########################
    #############################################################

                    ############################
                    ######## ESSENTIAL #########
                    ############################

# folder where you store all the images
    
imageFolder='/Volumes/bob/Stage/Lam/IMAGE/all_mosaique/'


# Name of the ASCII file where the list of input objects is stored.
# First column is assumed to be the ID. Sky or image coordinates
# must be included. The table can contain other columns, 
# which will be ignored here.

targetlist='/Users/basilehusquinet/Documents/Stage/L2/targetlist_Caitlin.txt'

# Type of coordinates (equatorial, galatic etc...)
frame='fk5'

# name of your all filters, 
# WARNING the name must appear in the name of the associated image, since we look in imageFolder for a file with this string in the name

filters=['HSC_i','HSC_z','UVISTA_Y','UVISTA_J','UVISTA_H','UVISTA_Ks','irac.1','irac.2']

# the HDU to be opened in the FITS file `imagefile`

hdu=0


                            ########################
                            ######### FLUX #########
                            ########################


# zero point of your image,
#(the list must have the same number as filters !!!!!!!!!!!!!)

zeroPoint=[27.,27.,30.,30.,30.,30.,21.58,21.58]

# gain for each image, used for the uncertainties. 
# If you have a infinite gain, note 0. by sextractor convention.
# (the list must have the same number as filters !!!!!!!!!!!!!)

gain=[18.5,12.1,0.,0.,0.,0.,0.,0.]

# radius of the aperture for each image in arcsecond. 
# It will be use to perform the aperture photometry.
#(the list must have the same number as filter!!!!!!!!!!!!!)

radius=[1.,1.,1.,1.,1.,1.,1.4,1.4]


#saveTbl: table with the aperture fluxes

saveTbl=True ; prefix='flux'


                            #########################
                            ######### STAMP #########
                            #########################


# size of the first stamp in arcsecond (first image in "filters"). All other stamps will be scaled accordingly

sizeStamp=12

#saveFits: stamps in fits file for each source, centered on the coordinate of your object
#with the sizeRef. 

#savePDF: Pdf file with all stamps

saveFits=True ; savePDF=True



                        ############################
                        ######### STACKING #########
                        ############################

#stacking: Pdf with plot of magnitude on fonction of the wavelength
stacking=True

# wavelenght of your filters
wavelenght=[7741., 8912., 10214.2, 12534.6, 16453.4, 21539.9, 36000., 45000.]

# radius transpose arcsec in pixel
radiusStack=np.zeros(np.size(filters))


    #############################################################
    ########################### BEGIN ###########################
    #############################################################


# Derive the time of execution
startTime = time.time()

# Recover the images and table
image, tbl=opens(imageFolder,targetlist,filters,frame=frame)


# Array storing all stamps (N filt x N sources)
cutout=list(np.zeros(np.size(image)))

# Look if a folder Stamp_fits/ exists (to save the stamps in .fits)
dirpath = 'Result/'
if not os.path.exists(dirpath): #to be super-cautious and prevent overwriting
    os.makedirs(dirpath)

# n=0 if working on the reference image, n=1 if other (use to compute stamp size)
n=0

for i in range(np.size(image)):        
    print(image[i])


    ########################
    #### OPEN THE IMAGE #### 
    ########################

    # open fits file 
    image_hdu=fits.open(imageFolder+image[i]) 
    
    # Recover header
    hdr=image_hdu[hdu].header                
    
    # Recover data 
    image_data=image_hdu[hdu].data
    
    # Recover WCS 
    wcs_orig = WCS(image_hdu[hdu].header)
    image_hdu.close()
            
    # Numpy array where the data and the sky coordinates wcs are saved
    nddata = NDData(data=image_data,wcs=wcs_orig)
    
    
    
    #########################
    ######### STAMP #########
    #########################
    

    # Check if needed
    if saveFits or savePDF or stacking:
        
        # Define the units
        size=sizeStamp*u.arcsec
        
        # Cut the image into stamps at the coordinates of objects 
        cutout[i]=[Cutout2D(nddata.data,tbl['skycoord'][j],size,wcs=nddata.wcs) for j in range(np.size(tbl))]
        
        # Transpose arcsec in pixel
        radiusStack[i]=(radius[i]/(abs(hdr['CD1_1'])*3600))

    
    ########################
    ######### FLUX #########
    ########################

    if saveTbl:    
    
        # derive the flux and magnitude
        fluxs=flux(nddata, tbl, zeroPoint[i], gain[i], radius[i])
        
        # Recover the data of flux and magnitude
        tbl['flux_'+filters[i]]=fluxs['flux']
        tbl['flux_err_'+filters[i]]=fluxs['flux_err']
        tbl['mag_'+filters[i]]=fluxs['mag']
        tbl['mag_err_'+filters[i]]=fluxs['mag_err']



    ###############################
    ############ SAVE #############
    ###############################


# tbl
if saveTbl: 
    # save the table with all data
    tbl.write(dirpath+prefix+'.txt',format='ascii.commented_header',overwrite=True)#,format='ascii.fixed_width_two_line',format='ascii.commented_header',overwrite=True)

# stamp in fits
if saveFits:
    # save stamp in fits file
    stampFits(cutout, tbl, filters)

# PDF
if savePDF:
    # save the pdf 
    PDF(cutout)

# stacking all image filters    
if stacking:
        
    stack(cutout, radiusStack, gain, zeroPoint, wavelenght)    

    
endTime=round(time.time()-startTime,0)

print ('execution time: '+str(endTime)+' sec')
    
    
    
    
    
 
    
    
    
    
    
    
    
    
    
