import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.utils import get_custom_objects

from interlacer import layers, utils
import motion_sim.diff_forward_model as diff_forward_model, motion_sim.multicoil_motion_simulator as multicoil_motion_simulator
import training.hypermodels as hypermodels
import training.losses as losses


def reconstruct_rss(img_arr):
    comp_img_arr = utils.join_reim_channels(img_arr)
    sq = K.square(K.abs(comp_img_arr))
    return K.sqrt(K.sum(sq, axis=3))


def get_multicoil_interlacer_model(
        input_size,
        kernel_size,
        num_features,
        num_convs,
        num_layers,
        num_coils,
        hyp_model=False,
        motinp_model=False,
        n_units=256,
        output_domain='FREQ'):
    """Interlacer model with residual convolutions.

    Returns a model that takes a frequency-space input (of shape (batch_size, n, n, 2)) and returns
    a frequency-space output of the same size, comprised of interlacer layers and with connections
    from the input to each layer. Handles variable input size, and crops to a 320x320 image at the end.

    Args:
      input_size(int): Tuple containing input shape, excluding batch size
      kernel_size(int): Dimension of each convolutional filter
      num_features(int): Number of features in each intermediate network layer
      num_convs(int): Number of convolutions per layer
      num_layers(int): Number of convolutional layers in model
      num_coils(int): Number of coils in k-space data
      hyp_model(Bool): Whether to use a hypernetwork controlled by motion params
      motinp_model(Bool): Whether to use motion parameters as additional inputs to the model
      n_units(int): Number of units in the intermediate layers of hypernetwork MLP

    Returns:
      model: Keras model comprised of num_layers core interlaced layers with specified nonlinearities

    """
    inputs = Input(input_size)    
    if (hyp_model):
        hyp_net = hypermodels.get_motion_hypernetwork(n_units=n_units)
        hyp_input = hyp_net.input
        hyp_tensor = hyp_net.output
        m_inputs = (inputs, hyp_input)
    elif(motinp_model):
        mot_input = Input(shape=[18])  
        m_inputs = (inputs, mot_input)
    else:
        hyp_tensor = None
        m_inputs = inputs

    l_inputs = m_inputs if hyp_model else inputs
    inp_conv = layers.BatchNormConv(
        num_features, 1, hyp_conv=hyp_model)(l_inputs)

    inp_img = utils.convert_channels_to_image_domain(inputs)
    inp_img_unshift = tf.signal.ifftshift(inp_img, axes=(1, 2))

    l_inputs = (inp_img_unshift, hyp_tensor) if hyp_model else inp_img_unshift
    inp_img_conv_unshift = layers.BatchNormConv(
        num_features, 1, hyp_conv=hyp_model)(l_inputs)

    inp_img_conv = tf.signal.ifftshift(inp_img_conv_unshift, axes=(1, 2))

    freq_in = inputs
    img_in = inp_img

    if(motinp_model):
        freq_in = Concatenate()([freq_in,tf.tile(tf.expand_dims(tf.expand_dims(mot_input,1),1),[1,tf.shape(inputs)[1],tf.shape(inputs)[2],1])])
        img_in = Concatenate()([img_in,tf.tile(tf.expand_dims(tf.expand_dims(mot_input,1),1),[1,tf.shape(inputs)[1],tf.shape(inputs)[2],1])])

    for i in range(num_layers):
        if(hyp_model):
            m_in = [img_in, freq_in, hyp_tensor]
        else:
            m_in = [img_in, freq_in]
        img_conv, k_conv = layers.Interlacer(
            num_features, kernel_size, num_convs, shift=True, hyp_conv = hyp_model)(m_in)

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
    def __init__(self, hyp_model, use_gt_params, *args, **kwargs):
        super(CustomModel, self).__init__(*args, **kwargs)
        self.hyp_model = hyp_model
        self.use_gt_params = use_gt_params
    
    @tf.function
    def compute_losses(self, data, training):        
        inputs, outputs = data
        
        k_true = outputs['k_true']
        angle_true = outputs['angle_true']
        num_pix_true = outputs['num_pix_true']
        k_corrupt = outputs['k_corrupt']
        k_corrupt.set_shape((None,None,None,44*2))

        model_outputs = self(inputs, training=training)      
        
        k_pred = model_outputs['k_pred']
        ssim_loss = tf.reduce_mean(losses.multicoil_ssim('FREQ', 44)(k_true, k_pred))
                    
        if(self.hyp_model and not self.use_gt_params):
            angle_pred = model_outputs['angles']
            num_pix_pred = model_outputs['num_pixes']
            
            rot_loss = 1 / 20.0 * tf.reduce_mean(tf.abs(angle_pred - angle_true), axis=(0, 1))
            trans_loss = 1 / 20.0 * tf.reduce_mean(tf.abs(num_pix_pred - num_pix_true),axis=(0,1,2))
        else:
            rot_loss = 0
            trans_loss = 0

        return ssim_loss, rot_loss, trans_loss

    @tf.function
    def train_step(self, data):
        with tf.GradientTape() as tape:
            ssim_loss, rot_loss, trans_loss = self.compute_losses(data, training=True)
            loss = ssim_loss
                
            if(self.hyp_model and not self.use_gt_params):
                loss += rot_loss + trans_loss

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        return {
            "loss": loss,
            "image_ssim_multicoil": ssim_loss,
            "rot": rot_loss,
            "trans": trans_loss}
    
    @tf.function
    def test_step(self, data, training=False):
        ssim_loss, rot_loss, trans_loss = self.compute_losses(data, training=False)
        loss = ssim_loss

        if(self.hyp_model and not self.use_gt_params):
            loss += rot_loss + trans_loss


        return {
            "loss": loss,
            "image_ssim_multicoil": ssim_loss,
            "rot": rot_loss,
            "trans": trans_loss}


def get_motion_estimation_model(
        input_size,
        kernel_size,
        num_features,
        num_convs,
        num_layers,
        num_coils,
        n_units=256,
        hyp_model=False,
        motinp_model=False,
        output_domain='FREQ',
        use_gt_params=False):

    inputs = Input(input_size, name = 'k_in')

    if(use_gt_params):
        angle_true = Input((6),name='angle_true')
        num_pix_true = Input((6, 2),name='num_pix_true')

    if(hyp_model or motinp_model):        
        if(not use_gt_params):
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
        else:
            motion_params = Concatenate()([angle_true,Flatten()(num_pix_true)])

        angle_pred = motion_params[:, :6]
        num_pix_pred = Reshape((6, 2))(motion_params[:, 6:])
    
    correction_model = get_multicoil_interlacer_model(
        input_size,
        kernel_size,
        num_features,
        num_convs,
        num_layers,
        num_coils,
        hyp_model=hyp_model,
        motinp_model=motinp_model,
        n_units=n_units,
        output_domain=output_domain)

    correction_model.compile()

    if(hyp_model or motinp_model):
        recon_output = correction_model((inputs, motion_params))
    else:
        recon_output = correction_model(inputs)
    
    inputs = {'k_in': inputs}
    outputs = {'k_pred': recon_output}
    
    if(use_gt_params):
        inputs['angles'] = angle_true
        inputs['num_pixes'] = num_pix_true
    elif(hyp_model or motinp_model):
        outputs['angles'] = angle_pred
        outputs['num_pixes'] = num_pix_pred
    
    model = CustomModel(hyp_model, use_gt_params, inputs=inputs, outputs=outputs)

    return model
