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
tf.app.flags.DEFINE_string('data_dir', 'data/', """Path to the data directory.""")
tf.app.flags.DEFINE_integer('num_classes', 2, """ Number of classes""")

# >5k example lesions total
tf.app.flags.DEFINE_integer('epoch_size', 10, """Batch 1""")
tf.app.flags.DEFINE_integer('batch_size', 10, """Number of images to process in a batch.""")

# Testing parameters
tf.app.flags.DEFINE_string('RunInfo', 'dice10_wce1/', """Unique file name for this training run""")
tf.app.flags.DEFINE_integer('GPU', 1, """Which GPU to use""")
tf.app.flags.DEFINE_string('test_files', 'Test', """Testing files""")

# Hyperparameters:
tf.app.flags.DEFINE_float('dropout_factor', 0.5, """ Keep probability""")
tf.app.flags.DEFINE_float('l2_gamma', 1e-3, """ The gamma value for regularization loss""")
tf.app.flags.DEFINE_float('moving_avg_decay', 0.999, """ The decay rate for the moving average tracker""")
tf.app.flags.DEFINE_float('loss_factor', 1.0, """Penalty for missing a class is this times more severe""")
tf.app.flags.DEFINE_integer('loss_class', 1, """For classes this and above, apply the above loss factor.""")

# Hyperparameters to control the optimizer
tf.app.flags.DEFINE_float('learning_rate', 1e-3, """Initial learning rate""")
tf.app.flags.DEFINE_float('beta1', 0.9, """ The beta 1 value for the adam optimizer""")
tf.app.flags.DEFINE_float('beta2', 0.999, """ The beta 1 value for the adam optimizer""")

# Define a custom training class
def test():


    # Makes this the default graph where all ops will be added
    # with tf.Graph().as_default(), tf.device('/cpu:0'):
    with tf.Graph().as_default(), tf.device('/gpu:' + str(FLAGS.GPU)):

        # Define phase of training
        phase_train = tf.placeholder(tf.bool)

        # Load the images and labels.
        data, iterator = network.inputs(training=False, skip=True)

        # Define input shape
        data['data'] = tf.reshape(data['data'], [FLAGS.batch_size, 40, 24, 40])

        # Perform the forward pass:
        logits, l2loss = network.forward_pass(data['data'], phase_train=phase_train)

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

        # Run once for all the saved checkpoints
        ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir + FLAGS.RunInfo)
        for checkpoint in ckpt.all_model_checkpoint_paths:

            # Allow memory placement growth
            config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
            config.gpu_options.allow_growth = True
            with tf.Session(config=config) as mon_sess:

                # Print run info
                print("*** Validation Run %s on GPU %s ****" % (FLAGS.RunInfo, FLAGS.GPU))

                # Initialize the variables
                mon_sess.run(var_init)

                # Initialize iterator
                mon_sess.run(iterator.initializer)\

                # Restore the model
                saver.restore(mon_sess, checkpoint)

                # Extract the epoch
                Epoch = checkpoint.split('/')[-1].split('_')[-1]

                # Initialize the step counter
                step = 0
                # Set the max step count
                max_steps = int(FLAGS.epoch_size / FLAGS.batch_size)

                # Tester instance
                sdt = SDT.SODTester(True, False)

                try:
                    while step < max_steps:

                        # Load some metrics for testing
                        y_pred, examples = mon_sess.run([softmax, data], feed_dict={phase_train: False})

                        # Increment step
                        step += 1

                except tf.errors.OutOfRangeError:
                    print('Done with Training - Epoch limit reached')

                finally:

                    # Testing
                    pred_map = sdt.return_binary_segmentation(y_pred, 0.45, 1, True)
                    print ('Epoch: %s, Best Epoch: %s (%.3f)' %(Epoch, best_epoch, best_MAE))
                    dice, mcc = sdt.calculate_segmentation_metrics(pred_map, examples['label_data'])

                    # Lets save runs that perform well
                    if mcc >= best_MAE:

                        # Save the checkpoint
                        print(" ---------------- SAVING THIS ONE %s", ckpt.model_checkpoint_path)

                        # # Define the filenames
                        # checkpoint_file = os.path.join('testing/' + FLAGS.RunInfo, ('Epoch_%s_DICE_%0.3f' % (Epoch, sdt.AUC)))
                        #
                        # # Save the checkpoint
                        # saver.save(mon_sess, checkpoint_file)
                        #
                        # # Save a new best MAE
                        # best_MAE = mcc
                        # best_epoch = Epoch

                        # # Delete prior screenshots
                        # if tf.gfile.Exists('testing/' + FLAGS.RunInfo + 'Screenshots/'):
                        #     tf.gfile.DeleteRecursively('testing/' + FLAGS.RunInfo + 'Screenshots/')
                        # tf.gfile.MakeDirs('testing/' + FLAGS.RunInfo + 'Screenshots/')

                        # # TODO: Plot all the images
                        # for i in range (FLAGS.batch_size):
                        #
                        #     file = ('testing/' + FLAGS.RunInfo + 'Screenshots/' + 'test_%s.gif' %i)
                        #     sdt.plot_img_and_mask3D(examples['data'][i], pred_map[i], examples['label_data'][i], file)

                    # Shut down the session
                    mon_sess.close()

            # Print divider
            print('-' * 70)


def main(argv=None):  # pylint: disable=unused-argument
    test()

if __name__ == '__main__':
    tf.app.run()
