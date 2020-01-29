""" Training the network on a single GPU """

from __future__ import absolute_import      # import multi line and Absolute/Relative
from __future__ import division             # change the division operator to output float if dividing two integers
from __future__ import print_function       # use the print function from python 3

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

sdl= SDL.SODLoader(str(Path.home()) + '/PycharmProjects/Datasets/CT_Chest_ILD/')
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
        best_MAE, best_epoch = 0.25, 0

        while True:

            # Allow memory placement growth
            config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
            config.gpu_options.allow_growth = True
            with tf.Session(config=config) as mon_sess:

                # Print run info
                print("*** Validation Run %s on GPU %s ****" % (FLAGS.RunInfo, FLAGS.GPU))

                # Retreive the checkpoint
                ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir+FLAGS.RunInfo)

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
                    print ('No checkpoint file found')
                    break

                # Initialize the step counter
                step = 0
                # Set the max step count
                max_steps = int(FLAGS.epoch_size / FLAGS.batch_size)

                # Tester instance
                sdt = SDT.SODTester(True, False)

                # Run inference
                y_pred, examples = mon_sess.run([softmax, data], feed_dict={phase_train: False})

                # Testing
                pred_map = sdt.return_binary_segmentation(y_pred, 0.45, 1, True)
                print('Epoch: %s, Best Epoch: %s (%.3f)' % (Epoch, best_epoch, best_MAE))
                dice, mcc = sdt.calculate_segmentation_metrics(pred_map, examples['label_data'])

                # Lets save runs that perform well
                if mcc >= best_MAE:

                    # Save the checkpoint
                    print(" ---------------- SAVING THIS ONE %s", ckpt.model_checkpoint_path)

                    # Define the filenames
                    checkpoint_file = os.path.join('testing/' + FLAGS.RunInfo, ('Epoch_%s_DICE_%0.3f' % (Epoch, sdt.AUC)))

                    # Save the checkpoint
                    saver.save(mon_sess, checkpoint_file)

                    # Save a new best MAE
                    best_MAE = mcc
                    best_epoch = Epoch

                    if FLAGS.gif:

                        # Delete prior screenshots
                        if tf.gfile.Exists('testing/' + FLAGS.RunInfo + 'Screenshots/'):
                            tf.gfile.DeleteRecursively('testing/' + FLAGS.RunInfo + 'Screenshots/')
                        tf.gfile.MakeDirs('testing/' + FLAGS.RunInfo + 'Screenshots/')

                        # Plot all images
                        for i in range(FLAGS.batch_size):
                            file = ('testing/' + FLAGS.RunInfo + 'Screenshots/' + 'test_%s.gif' % i)
                            sdt.plot_img_and_mask3D(examples['data'][i], pred_map[i], examples['label_data'][i], file)

                # Shut down the session
                mon_sess.close()

            # # Break if this is the final checkpoint
            # try:
            #     if int(Epoch) > 976 in Epoch: break
            # except:
            #     if '1000' in Epoch: break

            # Print divider
            print('-' * 70)

            # Otherwise check folder for changes
            filecheck = glob.glob(FLAGS.train_dir+FLAGS.RunInfo + '*')
            newfilec = filecheck

            # Sleep if no changes
            while filecheck == newfilec:

                # Sleep an amount of time proportional to the epoch size
                time.sleep(5)

                # Recheck the folder for changes
                newfilec = glob.glob(FLAGS.train_dir+FLAGS.RunInfo + '*')


def main(argv=None):  # pylint: disable=unused-argument
    time.sleep(FLAGS.sleep)
    if tf.gfile.Exists('testing/' + FLAGS.RunInfo):
        tf.gfile.DeleteRecursively('testing/' + FLAGS.RunInfo)
    tf.gfile.MakeDirs('testing/' + FLAGS.RunInfo)
    test()

if __name__ == '__main__':
    tf.app.run()
