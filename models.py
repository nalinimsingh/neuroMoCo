import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.utils import get_custom_objects

from interlacer import layers, utils
import diff_forward_model
import hypermodels
import losses
import multicoil_motion_simulator


def reconstruct_rss(img_arr):
    comp_img_arr = utils.join_reim_channels(img_arr)
    sq = K.square(K.abs(comp_img_arr))
    return K.sqrt(K.sum(sq, axis=3))


def get_multicoil_interlacer_model(
        input_size,
        nonlinearity,
        kernel_size,
        num_features,
        num_convs,
        num_layers,
        num_coils,
        hyp_model=False,
        output_domain='FREQ'):
    """Interlacer model with residual convolutions.

    Returns a model that takes a frequency-space input (of shape (batch_size, n, n, 2)) and returns
    a frequency-space output of the same size, comprised of interlacer layers and with connections
    from the input to each layer. Handles variable input size, and crops to a 320x320 image at the end.

    Args:
      input_size(int): Tuple containing input shape, excluding batch size
      nonlinearity(str): 'relu' or '3-piece'
      kernel_size(int): Dimension of each convolutional filter
      num_features(int): Number of features in each intermediate network layer
      num_convs(int): Number of convolutions per layer
      num_layers(int): Number of convolutional layers in model
      enforce_dc(Bool): Whether to paste in original acquired k-space lines in final output

    Returns:
      model: Keras model comprised of num_layers core interlaced layers with specified nonlinearities

    """
    inputs = Input(input_size)
    if (hyp_model):
        hyp_net = hypermodels.get_motion_hypernetwork()
        hyp_input = hyp_net.input
        hyp_tensor = hyp_net.output
        m_inputs = (inputs, hyp_input)
    else:
        hyp_tensor = None
        m_inputs = inputs

    inp_conv = layers.BatchNormConv(
        num_features, 1, hyp_conv=hyp_model)(m_inputs)

    inp_img = utils.convert_channels_to_image_domain(inputs)
    inp_img_unshift = tf.signal.ifftshift(inp_img, axes=(1, 2))

    l_inputs = (inp_img_unshift, hyp_tensor) if hyp_model else inp_img_unshift
    inp_img_conv_unshift = layers.BatchNormConv(
        num_features, 1, hyp_conv=hyp_model)(l_inputs)

    inp_img_conv = tf.signal.ifftshift(inp_img_conv_unshift, axes=(1, 2))

    freq_in = inputs
    img_in = inp_img

    for i in range(num_layers):
        img_conv, k_conv = layers.Interlacer(
            num_features, kernel_size, num_convs, shift=True)([img_in, freq_in])

        freq_in = k_conv + inp_conv
        img_in = img_conv + inp_img_conv

    if (output_domain == 'FREQ'):
        output = Conv2D(
            2 * num_coils,
            kernel_size,
            activation=None,
            padding='same',
            kernel_initializer='he_normal')(freq_in) + inputs
    elif (output_domain == 'IMAGE'):
        img_in_unshift = tf.signal.ifftshift(img_in, axes=(1, 2))
        output_unshift = Conv2D(
            2 * num_coils,
            kernel_size,
            activation='relu',
            padding='same',
            kernel_initializer='he_normal')(img_in) + inp_img_unshift
        output = tf.signal.ifftshift(output_unshift, axes=(1, 2))

    model = keras.models.Model(inputs=m_inputs, outputs=output)
    return model




class CustomModel(keras.Model):
    def compute_losses(self, data):
        (k_corrupt, mapses, order_kys, norms), (k_true, angle_true, num_pix_true) = data
        k_pred, angle_pred, num_pix_pred = self((k_corrupt, mapses, order_kys, norms), training=True)
        ssim_loss = tf.reduce_mean(losses.multicoil_ssim('FREQ', 44)(k_true, k_pred))

        imgs = tf.map_fn(diff_forward_model.rss_image_from_multicoil_k, k_pred)[..., 0]
        elems = [utils.split_reim_tensor(imgs),mapses,order_kys,angle_pred,num_pix_pred,norms]
        map_dfm = (lambda x: diff_forward_model.add_rotation_and_translations(x[0], x[1], x[2], x[3], x[4], x[5]))

        forward, _ = tf.map_fn(map_dfm, elems, fn_output_signature=(tf.float32, tf.float32))

        rot_loss = 1 / 20.0 * tf.reduce_mean(tf.abs(angle_pred - angle_true), axis=(0, 1))
        trans_loss = 1 / 20.0 * tf.reduce_mean(tf.abs(num_pix_pred - num_pix_true),axis=(0,1,2))

        dc_loss = 1e-9 * tf.reduce_sum(tf.square(tf.abs(utils.join_reim_channels(forward) - utils.join_reim_channels(k_corrupt))),axis=(0,1,2,3))

        return ssim_loss, dc_loss, rot_loss, trans_loss


    def train_step(self, data):
        with tf.GradientTape() as tape:
            ssim_loss, dc_loss, rot_loss, trans_loss = self.compute_losses(data)
            loss = ssim_loss + dc_loss + rot_loss + trans_loss

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        return {
            "loss": loss,
            "image_ssim_multicoil": ssim_loss,
            "dc": dc_loss,
            "rot": rot_loss,
            "trans": trans_loss}

    def test_step(self, data):
        ssim_loss, dc_loss, rot_loss, trans_loss = self.compute_losses(data)
        loss = ssim_loss + dc_loss + rot_loss + trans_loss

        return {
            "loss": loss,
            "image_ssim_multicoil": ssim_loss,
            "dc": dc_loss,
            "rot": rot_loss,
            "trans": trans_loss}


def get_motion_estimation_model(
        input_size,
        nonlinearity,
        kernel_size,
        num_features,
        num_convs,
        num_layers,
        num_coils,
        hyp_model=False,
        output_domain='FREQ',
        data_consistency=False):

    inputs = Input(input_size)

    if (data_consistency):
        maps = Input(input_size[:-1] + (int(input_size[-1] / 2),))
        order_ky = Input(input_size + (6,))
        angle_true = Input((6))
        num_pix_true = Input((6, 2))
        norms = Input((1))

    n = tf.shape(inputs)[1]
    inp_real = tf.expand_dims(inputs[:, :, :, 0], -1)
    inp_imag = tf.expand_dims(inputs[:, :, :, 1], -1)

    n_copies = int(num_features / 2)

    inp_conv = layers.BatchNormConv(
        num_features, 1, hyp_conv=False)(inputs)

    inp_img = utils.convert_channels_to_image_domain(inputs)
    inp_img_unshift = tf.signal.ifftshift(inp_img, axes=(1, 2))

    inp_img_conv_unshift = layers.BatchNormConv(
        num_features, 1, hyp_conv=False)(inp_img_unshift)

    inp_img_conv = tf.signal.ifftshift(inp_img_conv_unshift, axes=(1, 2))

    freq_in = inputs
    img_in = inp_img

    for i in range(3):
        i_inputs = (img_in, freq_in)
        img_conv, k_conv = layers.Interlacer(
            num_features, kernel_size, num_convs, hyp_conv=False, shift=True)(i_inputs)

        freq_in = k_conv + inp_conv
        img_in = img_conv + inp_img_conv

    pooled_conv_freq = GlobalAveragePooling2D()(
        layers.BatchNormConv(18, 1, hyp_conv=False)(freq_in))
    motion_params = Flatten()(pooled_conv_freq)
    for n in range(3):
        # Batch norm before the dense layers hurts performance
        motion_params = Dense(
            18, activation='relu', name='motion_param_dense_%d' %
            (n + 1))(motion_params)

    angle_pred = motion_params[:, :6]
    num_pix_pred = Reshape((6, 2))(motion_params[:, 6:])

    correction_model = get_multicoil_interlacer_model(
        input_size,
        nonlinearity,
        kernel_size,
        num_features,
        num_convs,
        num_layers,
        num_coils,
        hyp_model=hyp_model,
        output_domain=output_domain)

    correction_model.compile()

    output = correction_model((inputs, motion_params))

    if(data_consistency):
        model = CustomModel(inputs=(inputs, maps, order_ky, norms), outputs=(output, angle_pred, num_pix_pred))
    else:
        model = keras.models.Model(inputs=inputs, outputs=output)

    return model


def get_multicoil_conv_model(input_size, nonlinearity, kernel_size,
                             num_features, num_layers, num_coils): 
    """Alternating model with residual
        convolutions.

    Returns a model that takes a frequency-space input (of shape (batch_size, n,
    n, num_coils*2)) and returns a frequency-space output of the same size,
    comprised of alternating frequency- and image-space convolutional layers and
    with connections from the input to each layer.

    Args: input_size(int): Tuple containing input shape, excluding batch size
    nonlinearity(str): 'relu' or '3-piece' kernel_size(int): Dimension of each
    convolutional filter num_features(int): Number of features in each
    intermediate network layer num_layers(int): Number of convolutional layers
    in model num_coils(int): Number of coils in final output

    Returns: model: Keras model comprised of num_layers alternating image- and
    frequency-space convolutional layers with specified nonlinearities

    """
    inputs = Input(input_size)

    inp_conv = layers.BatchNormConv(num_features, 1)(inputs)

    prev_layer = inputs

    for i in range(num_layers):
        conv = layers.BatchNormConv(num_features,
                                    kernel_size)(prev_layer) 
        nonlinear = layers.get_nonlinear_layer(nonlinearity)(conv) 
        prev_layer = nonlinear + inp_conv

    output = Conv2D(
        num_coils * 2,
        kernel_size,
        activation=None,
        padding='same',
        kernel_initializer='he_normal')(prev_layer) + inputs

    model = keras.models.Model(inputs=inputs, outputs=output) 
    return model
