""" Training the network on a single GPU """

from __future__ import absolute_import  # import multi line and Absolute/Relative
from __future__ import division  # change the division operator to output float if dividing two integers
from __future__ import print_function  # use the print function from python 3

import os
import time
import numpy as np

import DTIModel as network
import tensorflow as tf
import SODTester as SDT
import SODLoader as SDL
import SOD_Display as SDD
import glob
from pathlib import Path

import cv2, imageio
import matplotlib.pyplot as plt
import sklearn

sdl = SDL.SODLoader(str(Path.home()) + '/PycharmProjects/Datasets/CT_Chest_ILD/')
sdd = SDD.SOD_Display()

_author_ = 'Simi'

# Define the FLAGS class to hold our immutable global variables
FLAGS = tf.app.flags.FLAGS

# Define some of the immutable variables
tf.app.flags.DEFINE_string('train_dir', 'training/', """Directory to write event logs and save checkpoint files""")
tf.app.flags.DEFINE_string('data_dir', 'data/test/', """Path to the data directory.""")
tf.app.flags.DEFINE_integer('num_classes', 2, """ Number of classes""")

# >5k example lesions total
tf.app.flags.DEFINE_integer('epoch_size', 62, """Batch 1""")
tf.app.flags.DEFINE_integer('batch_size', 62, """Number of images to process in a batch.""")

# Testing parameters
tf.app.flags.DEFINE_string('RunInfo', 'New_data1/', """Unique file name for this training run""")
tf.app.flags.DEFINE_integer('GPU', 1, """Which GPU to use""")
tf.app.flags.DEFINE_string('test_files', 'Test', """Testing files""")
tf.app.flags.DEFINE_integer('sleep', 0, """ Time to sleep before starting test""")
tf.app.flags.DEFINE_integer('gifs', 0, """ save gifs or not""")

# Hyperparameters:
tf.app.flags.DEFINE_float('dropout_factor', 0.5, """ Keep probability""")
tf.app.flags.DEFINE_float('moving_avg_decay', 0.999, """ The decay rate for the moving average tracker""")


# Define a custom training class
def test():
    # Makes this the default graph where all ops will be added
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        # with tf.Graph().as_default(), tf.device('/gpu:' + str(FLAGS.GPU)):

        # Define phase of training
        phase_train = tf.placeholder(tf.bool)

        # Load the images and labels.
        iterator = network.inputs(training=False, skip=True)
        data = iterator.get_next()

        # Define input shape
        data['data'] = tf.reshape(data['data'], [FLAGS.batch_size, 40, 24, 40])

        # Perform the forward pass:
        logits = network.forward_pass(data['data'], phase_train=phase_train)

        # Retreive softmax
        softmax = tf.nn.softmax(logits)

        # Summary images
        imeg = int(FLAGS.batch_size / 2)
        tf.summary.image('Labels_Test', tf.reshape(tf.cast(data['label_data'][imeg, 20, ...], tf.float32), shape=[1, 24, 40, 1]), 2)
        tf.summary.image('Logits_Test', tf.reshape(logits[imeg, 20, :, :, 1], shape=[1, 24, 40, 1]), 2)

        # Initialize variables operation
        var_init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        # Restore moving average of the variables
        var_ema = tf.train.ExponentialMovingAverage(FLAGS.moving_avg_decay)

        # Define variables to restore
        var_restore = var_ema.variables_to_restore()

        # Initialize the saver
        saver = tf.train.Saver(var_restore, max_to_keep=3)

        # Trackers for best performers
        best_score, best_epoch = 0.25, 0

        # Allow memory placement growth
        config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as mon_sess:

            # Retreive the checkpoint
            ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir + FLAGS.RunInfo)

            # Initialize the variables
            mon_sess.run(var_init)

            # Initialize iterator
            mon_sess.run(iterator.initializer)

            if ckpt and ckpt.model_checkpoint_path:

                # Restore the model
                saver.restore(mon_sess, ckpt.model_checkpoint_path)

                # Extract the epoch
                Epoch = ckpt.model_checkpoint_path.split('/')[-1].split('_')[-1]

            else:
                print('No checkpoint file found')
                return

            # Tester instance
            sdt = SDT.SODTester(True, False)

            # Run inference
            y_pred, examples = mon_sess.run([softmax, data], feed_dict={phase_train: False})

            # Testing
            pred_map = sdt.return_binary_segmentation(y_pred, 0.45, 1, True)

            # Code to save volumes
            for i in range(FLAGS.batch_size):
                # Data was cropped from: Z//2, Ycn + 7 and Xcn to a window Z//2, 12, 20 big then pad resized to 40, 24, 40
                save_vol = pred_map[i].astype(np.uint8)
                save_vol = sdl.largest_blob(save_vol)[0].astype(np.uint8)

                # Retreive the initial measurements
                osi = np.asarray([int(examples['Zinit'][i]), int(examples['Yinit'][i]), int(examples['Xinit'][i])])
                ocn = np.asarray([int(examples['Zinit'][i]) // 2, int(examples['Ycn'][i]), int(examples['Xcn'][i])])
                cn = np.asarray(save_vol.shape) // 2

                # Crop resize from 40, 24, 40 to z, 24, 40.
                save_vol = sdl.crop_data(save_vol, cn, [osi[0] // 2, 12, 20])[0]

                # Pad resize from z//2, 12, 20 to original size from center points Ycn - 7
                xpad = ((osi[2] - save_vol.shape[2]) // 2) + (ocn[2] - (osi[2] // 2))
                xpad2 = ((osi[2] - save_vol.shape[2]) // 2) - (ocn[2] - (osi[2] // 2))
                ypad = ((osi[1] - save_vol.shape[1]) // 2) + (ocn[1] - (osi[1] // 2))
                ypad2 = ((osi[1] - save_vol.shape[1]) // 2) - (ocn[1] - (osi[1] // 2))
                zpad = (osi[0] - save_vol.shape[0]) // 2

                # Perform the pad
                save_vol = np.pad(save_vol, ((zpad, zpad), (ypad, ypad2), (xpad, xpad2)), 'constant')

                # Final resize to actual size (accounts for odd numbers)
                save_vol = sdl.pad_resize(save_vol, osi)

                file = ('testing/' + FLAGS.RunInfo + 'Screenshots/' + 'Pred_%s.nii' % examples['acc'][i].decode('utf-8'))
                sdl.save_image(save_vol, file)

                # TODO: Testing
                # if i<=6: sdd.display_volume(save_vol)

                # Garbage
                del save_vol

            # Shut down the session
            mon_sess.close()
            plt.show()


def main(argv=None):
    test()


if __name__ == '__main__':
    tf.app.run()
