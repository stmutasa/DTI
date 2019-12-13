from glob import glob
import pydicom as dicom
import SODLoader as SDL
import SOD_Display as SDD

sdl = SDL.SODLoader('/data')
sdd = SDD.SOD_Display()
import numpy as np


def get_b0():
    """
    Function to get the B0 sequence from Siemens DTI scans. The B0 comes in a tiled 6x6 image of 128x128
    For SIEMENS MRI:
    0019; 100A;  Number Of Images In Mosaic
    0019; 100B;  Slice Measurement Duration
    0019; 100C;  B_value
    0019; 100D; Diffusion Directionality
    0019; 100E; Diffusion Gradient Direction
    0019; 100F;  Gradient Mode
    0019; 1027;  B_matrix
    0019; 1028;  Bandwidth Per Pixel Phase Encode

    For GE MRI:
    Software version 12.0 (0018; 1020)
    0043; 1039; first string is b value
    0019; 10BB; x gradient;
    0019; 10BC; y gradient;
    0019; 10BC; z gradient;

    For Philips MRI
     Software version 3.2.1 (0018; 1020)
    0018; 9087; b value
    0018; 9089; Diffusion direction
    """

    # Path to new Siemens DTI
    path = '/home/stmutasa/PycharmProjects/Datasets/DTI Data/CHOP Data'

    # Get the files
    files = sdl.retreive_filelist('dcm', True, path)

    # Variables to track
    index, length, lastpt, num = 0, len(files), '000', 1

    print('Looping through %s files!' % length)
    for dcm in files:

        # Retreive the files
        try:
            info = dicom.read_file(dcm)
        except:
            continue

        # Get the patient info
        acc = info['StudyID'].value
        ser = info['SeriesDescription'].value
        file_index = dcm.split('/')[-3]

        # If this is the same patient,increment index
        if acc == lastpt:
            num += 1
        else:
            lastpt = acc

        # Work only on DTI sequences
        if 'DTI' not in ser: continue

        try:

            # Get the b-Value from the element tag
            bvalue = info[0x0019, 0x100C].value.decode('utf-8')

            # Skip B600, only save B0
            if '600' in bvalue: continue

            # Get the image pixels
            image = info.pixel_array
            shape = info['AcquisitionMatrix'].value
            print('Acc: %s, Ser: %s, B-Value: %s ******* Shape: ' % (acc, ser, bvalue), shape)

            # Make the volume
            array_size, slice = 6, 0
            volume = np.zeros([36, 128, 128], np.int16)
            for y in range(array_size):
                for x in range(array_size):
                    # Calculate the y and x extent of the patch
                    ya, yb = y * 128, y * 128 + 128
                    xa, xb = x * 128, x * 128 + 128
                    patch = image[ya:yb, xa:xb]

                    # Add patch to volume
                    volume[slice] = patch
                    slice += 1

            # Save the volume as nifti
            save_file = ('b0/%s_%s_%s.nii.gz' % (file_index, num, acc))
            sdl.save_volume(volume, save_file)

            # Increment counter
            index += 1
            if index % (length // 20) == 0: print(' %s files of %s complete (%.2f%%)' % (index, length, 100 * index / length))

            # Garbage
            del image, info
        except:
            del info
            index += 1
            if index % (length // 20) == 0: print(' %s files of %s complete (%.2f%%)' % (index, length, 100 * index / length))
            continue


get_b0()
