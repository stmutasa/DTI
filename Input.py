"""
Load and preprocess the files to a protobuff
"""

import numpy as np
import tensorflow as tf
import SODLoader as SDL
import SOD_Display as SDD

from pathlib import Path
from random import shuffle
import matplotlib.pyplot as plt
import os

# TODO: Testing
import mclahe as mc

# Define the flags class for variables
FLAGS = tf.app.flags.FLAGS

# Define the data directory to use
# home_dir = str(Path.home()) + '/PycharmProjects/Datasets/DTI_Data/Data Christian/Complete imaging HRNB 3.27.18/DTI selected/'
home_dir = str(Path.home()) + '/PycharmProjects/Datasets/DTI_Data/DTI/'

sdl = SDL.SODLoader(data_root=home_dir)
sdd = SDD.SOD_Display()

# For loading the files for a 2.5 D network
def pre_proc():

    """
    Loads the data into tfrecords
    :param dims:
    :return:
    """

    # First retreive the filenames
    filenames = sdl.retreive_filelist('nii.gz', path=home_dir, include_subfolders=False)
    filenames = [x for x in filenames if '.csv' not in x]
    filenames = [x for x in filenames if '-label' in x]
    shuffle(filenames)

    # retrieve the labels
    label_file = sdl.retreive_filelist('csv', path=home_dir, include_subfolders=True)[0]
    labels = sdl.load_CSV_Dict('PT', label_file)

    # global variables
    index, pts, per = 0, 0, 0
    data, track = {}, {}
    display, failures = [], [0, 0, 0]

    # Loop through all the patients
    for file in filenames:

        # Filenames
        dirname = os.path.dirname(file)
        basename = os.path.basename(file)

        # Patient info
        pt_id = basename.split('_')[0]
        image_file = dirname + '/' + basename.split('-')[0] + '.nii.gz'
        seq_id = basename.split('-')[0]

        # TODO: Load info from labels.csv here

        # Now load the volumes
        try:
            volume = np.squeeze(sdl.load_NIFTY(image_file))
            volume = np.squeeze(volume[..., 0])
            segments = np.squeeze(sdl.load_NIFTY(file))
        except:
            print ('Unable to load: ', file, '\n')
            failures[2] +=1
            continue

        # Swap x and Y axes
        volume, segments = np.swapaxes(volume, 2, 0).astype(np.int16), np.swapaxes(segments, 1, 2)

        """ 
            There is too much pixel data right now for a 3D network. But, the physis is actually consistently 
            in the middle of the image and the femoral physis is slightly superiorly located. 
            Crop the volumes starting at about 60, 60, 20 in x and 10 in y, none in z
        """

        # Crop
        volume, _, _ = sdl.crop_data(volume, [volume.shape[0]//2, 60, 60], [volume.shape[0]//2, 12, 20])
        segments, _, _ = sdl.crop_data(segments, [segments.shape[0] // 2, 60, 60], [segments.shape[0] // 2, 12, 20])

        # Resize (pad)
        volume = sdl.pad_resize(volume, [40, 24, 40])
        segments = sdl.pad_resize(segments, [40, 24, 40])

        # Normalize the MRI with contrast adaptive histogram normalization
        volume = mc.mclahe(volume)

        # Save the volume and segments
        data[index] = {'data': volume, 'label_data': segments, 'acc': seq_id, 'file': file, 'mrn': pt_id,
                       'dx': volume.shape[2], 'dy': volume.shape[1], 'dz': volume.shape[0]}

        # Finished with this patient
        index += 1
        pts += 1

        # Garbage collection
        del volume, segments

    # All patients done, print the summary message
    print('%s Patients processed, %s failed[No label, Label out of range, Failed load] %s' % (pts, sum(failures), failures))

    # Now create a final protocol buffer
    print('Creating protocol buffer')
    if data:
        sdl.save_dict_filetypes(data[0])
        sdl.save_tfrecords(data, 2, test_size=2, file_root='data/Vols')
        print('%s patients complete, %s volumes saved' % (pts, index))
        del data, display


# Load the protobuf
def load_protobuf(training=True):

    """
    Loads the protocol buffer into a form to send to shuffle
    """

    # Define filenames
    if training:
        all_files = sdl.retreive_filelist('tfrecords', False, path=FLAGS.data_dir)
        filenames = [x for x in all_files if FLAGS.test_files not in x]
    else:
        all_files = sdl.retreive_filelist('tfrecords', False, path=FLAGS.data_dir)
        filenames = [x for x in all_files if FLAGS.test_files in x]

    print('******** Loading Files: ', filenames)

    # Create a dataset from the protobuf
    dataset = tf.data.TFRecordDataset(filenames)

    # Shuffle the entire dataset then create a batch
    if training: dataset = dataset.shuffle(buffer_size=FLAGS.epoch_size)

    # Load the tfrecords into the dataset with the first map call
    _records_call = lambda dataset: \
        sdl.load_tfrecords(dataset, [40, 24, 40], tf.float32,
                           segments='label_data', segments_dtype=tf.int16, segments_shape=[40, 24, 40])

    # Parse the record into tensors
    dataset = dataset.map(_records_call, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Fuse the mapping and batching
    scope = 'data_augmentation' if training else 'input'
    with tf.name_scope(scope):

        # Map the data set
        dataset = dataset.map(DataPreprocessor(training), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Batch the dataset and drop remainder. Can try batch before map if map is small
    dataset = dataset.batch(FLAGS.batch_size, drop_remainder=True)

    # cache and Prefetch
    dataset = dataset.cache()
    dataset = dataset.prefetch(buffer_size=FLAGS.batch_size)

    # Repeat input indefinitely
    dataset = dataset.repeat()

    # Make an initializable iterator
    iterator = dataset.make_initializable_iterator()

    # Retreive the batch
    examples = iterator.get_next()

    # Return data as a dictionary
    return examples, iterator


class DataPreprocessor(object):

    # Applies transformations to dataset

  def __init__(self, distords):
    self._distords = distords

  def __call__(self, record):

    """Process img for training or eval."""
    image = record['data']

    if self._distords:  # Training

        # Data Augmentation ------------------

        # For noise, first randomly determine how 'noisy' this study will be
        T_noise = tf.random_uniform([1], 0, 0.02)

        # Create a poisson noise array
        noise = tf.random_uniform(shape=[40, 24, 40], minval=-T_noise, maxval=T_noise)

        # Add the poisson noise
        image = tf.add(image, tf.cast(noise, tf.float32))

    # Make record image
    record['data'] = image

    return record

# pre_proc()
