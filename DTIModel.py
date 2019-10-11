# Defines and builds our network
#    Computes input images and labels using inputs() or distorted inputs ()
#    Computes inference on the models (forward pass) using inference()
#    Computes the total loss using loss()
#    Performs the backprop using train()

from __future__ import absolute_import  # import multi line and Absolute/Relative
from __future__ import division  # change the division operator to output float if dividing two integers
from __future__ import print_function  # use the print function from python 3

_author_ = 'simi'

import tensorflow as tf
import Input
import SODNetwork as SDN

# Define the FLAGS class to hold our immutable global variables
FLAGS = tf.app.flags.FLAGS

# Retreive helper function object
sdn = SDN.SODMatrix()
sdloss = SDN.SODLoss(2)

def forward_pass(images, phase_train):

    """
    Train a 3 dimensional U-network
    :param images: input images, [batch, z, y, x]
    :param phase_train: True if this is the training phase
    :return: L2 Loss and Logits
    """

    # Initial kernel size
    K = 4
    images = tf.expand_dims(images, -1) # batch x 32 x 24 x 40

    # 3D Unet here
    conv1 = sdn.convolution_3d('Conv1a', images, 3, K, S=1, phase_train=phase_train)

    # 16x12x20 begin residuals
    conv2 = sdn.convolution_3d('Conv1ds', conv1, 3, K*2, S=2, phase_train=phase_train)
    conv2 = sdn.residual_layer_3d('Conv2a', conv2, 3, K*2, S=1, phase_train=phase_train)

    # 8x6x10 Begin inception
    conv3 = sdn.convolution_3d('Conv2ds', conv2, 3, K*4, S=2, phase_train=phase_train)
    conv3 = sdn.residual_layer_3d('Conv3a', conv3, 3, K * 4, S=1, phase_train=phase_train)
    conv3 = sdn.residual_layer_3d('Conv3b', conv3, 3, K * 4, S=1, phase_train=phase_train)

    # Bottom: 4x3x5
    conv = sdn.convolution_3d('Conv3ds', conv3, 3, K*8, S=2, phase_train=phase_train)
    conv = sdn.inception_layer_3d('Conv4a', conv, K*8, S=1, phase_train=phase_train)
    conv = sdn.residual_layer_3d('Conv4b', conv, 3, K * 8, S=1, phase_train=phase_train)
    conv = sdn.inception_layer_3d('Conv4c', conv, K * 8, S=1, phase_train=phase_train)
    conv = sdn.residual_layer_3d('Conv4d', conv, 3, K * 8, S=1, phase_train=phase_train)
    
    # Upsample
    conv = sdn.deconvolution_3d('Dconv4', conv, 3, K*4, 2, phase_train=phase_train, concat=False, concat_var=conv3)
    conv = sdn.residual_layer_3d('Dconv3a', conv, 3, K * 4, S=1, phase_train=phase_train)

    conv = sdn.deconvolution_3d('Dconv3', conv, 3, K * 2, 2, phase_train=phase_train, concat=False, concat_var=conv2)
    conv = sdn.residual_layer_3d('Dconv2a', conv, 3, K * 2, S=1, phase_train=phase_train)

    conv = sdn.deconvolution_3d('Dconv2', conv, 3, K, 2, phase_train=phase_train, concat=False, concat_var=conv1)
    conv = sdn.convolution_3d('Dconv1a', conv, 3, K, S=1, phase_train=phase_train)
    Logits = sdn.convolution_3d('Logits', conv, 1, FLAGS.num_classes, S=1, phase_train=phase_train, BN=False, relu=False, bias=False)

    # Retreive the weights collection
    weights = tf.get_collection('weights')

    # Sum the losses
    L2_loss = tf.multiply(tf.add_n([tf.nn.l2_loss(v) for v in weights]), FLAGS.l2_gamma)

    # Add it to the collection
    tf.add_to_collection('losses', L2_loss)

    # Activation summary
    tf.summary.scalar('L2_Loss', L2_loss)

    return Logits, L2_loss  # Return whatever the name of the final logits variable is


def total_loss(logits_tmp, labels_tmp, loss_type='COMBINED'):

    """
    Loss function.
    :param logits: Raw logits - batchx32x24x40x2
    :param labels: The labels - batchx
    :param type: Type of loss
    :return:
    """

    # reduce dimensionality
    labels, logits = tf.squeeze(labels_tmp), tf.squeeze(logits_tmp)

    # Summary images
    imeg = int(FLAGS.batch_size / 2)
    tf.summary.image('Labels', tf.reshape(tf.cast(labels[imeg, 20, ...], tf.float32), shape=[1, 24, 40, 1]), 2)
    tf.summary.image('Logits', tf.reshape(logits_tmp[imeg, 20, :, :, 1], shape=[1, 24, 40, 1]), 2)

    # Make labels one hot
    labels = tf.cast(labels, tf.uint8)
    labels = tf.cast(tf.one_hot(labels, depth=FLAGS.num_classes, dtype=tf.uint8), tf.float32)

    # Flatten
    logits = tf.reshape(logits, [-1, FLAGS.num_classes])
    labels = tf.cast(tf.reshape(labels, [-1, FLAGS.num_classes]), tf.float32)

    if loss_type == 'DICE':

        # Get the generalized DICE loss
        loss = sdloss.dice(logits, labels)
        loss = tf.reduce_mean(loss)
        tf.summary.scalar('Dice Loss', loss)

    elif loss_type == 'WASS_DICE':

        # Get the generalized DICE loss
        loss = sdloss.generalised_wasserstein_dice_loss(labels, logits)
        loss = tf.reduce_mean(loss)
        tf.summary.scalar('WassersteinDice Loss', loss)

    elif loss_type == 'WCE':

        # Weighted CE, beta: > 1 decreases false negatives, <1 decreases false positives
        loss = sdloss.weighted_cross_entropy(logits, labels, beta=1)
        loss = tf.reduce_mean(loss)
        tf.summary.scalar('Cross Entropy Loss', loss)

    else:

        # Combine weighted cross entropy and DICe
        wce = sdloss.weighted_cross_entropy(logits, labels, 1)
        wce = tf.reduce_mean(wce)
        dice = sdloss.dice(logits, labels)
        dice = tf.reduce_mean(dice)

        # Add the losses with a weighting for each
        loss = wce*1 + dice*10

        # Output the summary of the MSE and MAE
        tf.summary.scalar('Cross Entropy Loss', wce)
        tf.summary.scalar('Dice Loss', dice)

    # Total loss
    tf.summary.scalar('Total loss', loss)

    # Add these losses to the collection
    tf.add_to_collection('losses', loss)

    return loss


def backward_pass(total_loss):
    """ This function performs our backward pass and updates our gradients
    Args:
        total_loss is the summed loss caculated above
        global_step1 is the number of training steps we've done to this point, useful to implement learning rate decay
    Returns:
        train_op: operation for training"""

    # Get the tensor that keeps track of step in this graph or create one if not there
    global_step = tf.contrib.framework.get_or_create_global_step()

    # Print summary of total loss
    tf.summary.scalar('Total_Loss', total_loss)

    # Decay the learning rate
    dk_steps = int((FLAGS.epoch_size / FLAGS.batch_size) * 150)
    lr_decayed = tf.train.cosine_decay_restarts(FLAGS.learning_rate, global_step, dk_steps)

    # Compute the gradients. NAdam optimizer came in tensorflow 1.2
    opt = tf.contrib.opt.NadamOptimizer(learning_rate=lr_decayed, beta1=FLAGS.beta1,
                                        beta2=FLAGS.beta2, epsilon=0.1)

    # Compute the gradients
    gradients = opt.compute_gradients(total_loss)

    # clip the gradients
    #gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients]

    # Apply the gradients
    train_op = opt.apply_gradients(gradients, global_step, name='train')

    # Add histograms for the trainable variables. i.e. the collection of variables created with Trainable=True
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    # Maintain average weights to smooth out training
    variable_averages = tf.train.ExponentialMovingAverage(FLAGS.moving_avg_decay, global_step)

    # Applies the average to the variables in the trainable ops collection
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([train_op, variable_averages_op]):  # Wait until we apply the gradients
        dummy_op = tf.no_op(name='train')  # Does nothing. placeholder to control the execution of the graph

    return dummy_op


def inputs(training=True, skip=True):

    """
    Loads the inputs
    :param filenames: Filenames placeholder
    :param training: if training phase
    :param skip: Skip generating tfrecords if already done
    :return:
    """

    # To Do: Skip part 1 and 2 if the protobuff already exists
    if not skip:
        Input.pre_proc()

    else:
        print('-------------------------Previously saved records found! Loading...')

    return Input.load_protobuf(training)


"""
Segmentation loss functions
"""

