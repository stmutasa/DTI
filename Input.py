"""
Load and preprocess the files to a protobuff
"""

import numpy as np
import tensorflow as tf
import SODLoader as SDL
import SOD_Display as SDD
from random import shuffle

from pathlib import Path
import os

import mclahe as mc

# Define the flags class for variables
FLAGS = tf.app.flags.FLAGS

# Define the data directory to use
home_dir = str(Path.home()) + '/PycharmProjects/Datasets/DTI_Data/Labeled/'

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
    filenames = sdl.retreive_filelist('nii.gz', path=home_dir, include_subfolders=True)
    filenames += sdl.retreive_filelist('nii', path=home_dir, include_subfolders=True)
    filenames = [x for x in filenames if '-label' in x]

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

        # Now load the volumes
        try:
            volume = np.squeeze(sdl.load_NIFTY(image_file))
            if int(pt_id) <= 299: volume = np.squeeze(volume[..., 0])
            segments = np.squeeze(sdl.load_NIFTY(file))
        except:
            print('Unable to load: ', file, '\n')
            failures[2] += 1
            continue

        # Swap x and Y axes
        if int(pt_id) <= 299:
            volume, segments = np.swapaxes(volume, 2, 0).astype(np.int16), np.swapaxes(segments, 1, 2)
        else:
            volume, segments = np.swapaxes(volume, 1, 0).astype(np.int16), np.swapaxes(segments, 1, 0)
            volume, segments = np.swapaxes(volume, 1, 2).astype(np.int16), np.swapaxes(segments, 1, 2)

        """ 
            There is too much pixel data right now for a 3D network. But, the physis is actually consistently 
            in the middle of the image and the femoral physis is slightly superiorly located. 
            Crop the volumes starting at about 60, 60, 20 in x and 10 in y, none in z
        """

        # Save the initial z dimensions
        Zinit, Yinit, Xinit = volume.shape

        # Crop
        _, cn = sdl.largest_blob(segments)
        # volume, _, _ = sdl.crop_data(volume, [volume.shape[0]//2, 60, 60], [volume.shape[0]//2, 12, 20])
        volume, _, _ = sdl.crop_data(volume, [volume.shape[0] // 2, cn[1] + 7, cn[2]], [volume.shape[0] // 2, 12, 20])
        segments, _, _ = sdl.crop_data(segments, [segments.shape[0] // 2, cn[1] + 7, cn[2]], [segments.shape[0] // 2, 12, 20])

        # Resize (pad)
        volume = sdl.pad_resize(volume, [40, 24, 40])
        segments = sdl.pad_resize(segments, [40, 24, 40])

        # Normalize the MRI with contrast adaptive histogram normalization
        volume = mc.mclahe(volume)

        # Save the volume and segments
        # Data was cropped from: Z//2, Ycn + 7 and Xcn to a window Z//2, 12, 20 big then resized to 40, 24, 40
        data[index] = {'data': volume, 'label_data': segments, 'acc': seq_id, 'file': file, 'mrn': pt_id,
                       'Zinit': Zinit, 'Yinit': Yinit, 'Xinit': Xinit, 'Ycn': cn[1], 'Xcn': cn[2]}

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
        sdl.save_segregated_tfrecords(4, data, 'mrn', 'data/Vols')
        print('%s patients complete, %s volumes saved' % (pts, index))
        del data, display


# Load the protobuf
def load_protobuf(training=True):

    """
    Loads the protocol buffer into a form to send to shuffle
    """

    # Load tfrecords with parallel interleave if training
    if training:
        filenames = sdl.retreive_filelist('tfrecords', False, path=FLAGS.data_dir)
        files = tf.data.Dataset.list_files(os.path.join(FLAGS.data_dir, '*.tfrecords'))
        dataset = files.interleave(tf.data.TFRecordDataset, cycle_length=len(filenames),
                                   num_parallel_calls=tf.data.experimental.AUTOTUNE)
        print('******** Loading Files: ', filenames)
    else:
        files = sdl.retreive_filelist('tfrecords', False, path=FLAGS.data_dir)
        dataset = tf.data.TFRecordDataset(files, num_parallel_reads=1)
        print('******** Loading Files: ', files)

    # Repeat -> Shuffle -> Map -> Batch -> Prefetch
    dataset = dataset.repeat()

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

    # Prefetch
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    # Make an initializable iterator
    iterator = dataset.make_initializable_iterator()

    # Return data as a dictionary
    return iterator


class DataPreprocessor(object):

    # Applies transformations to dataset

  def __init__(self, distords):

    self._distords = distords

  def __call__(self, record):

    """Process img for training or eval."""
    image = record['data']
    labels = record['label_data']

    if self._distords:  # Training

        # Data Augmentation ------------------

        # For noise, first randomly determine how 'noisy' this study will be
        T_noise = tf.random.uniform([], 0, 0.04)

        # Create a poisson noise array
        noise = tf.random.uniform(shape=[40, 24, 40], minval=-T_noise, maxval=T_noise)

        # Perform random rotation. Use nearest neighbor for labels because we need 1 or 0 values
        img, lbl = [], []

        # Random angle for rotation
        angle = tf.random.uniform([], -0.30, 0.30)

        # Perform the rotation
        for z in range(40):
            img.append(tf.contrib.image.rotate(image[z], angle, 'BILINEAR'))
            lbl.append(tf.contrib.image.rotate(tf.cast(labels[z], tf.float32), angle, 'NEAREST'))

        image, labels = tf.stack(img), tf.stack(lbl)

        # Add the poisson noise
        image = tf.add(image, tf.cast(noise, tf.float32))

    else: # Testing

        pass

    # Make record image
    record['data'] = image
    record['label_data'] = labels

    return record