import math
import sys

import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.keras import backend as K
import voxelmorph as vxm

from interlacer import utils

def simulate_multicoil_k(image, maps):
    """
    image: (x,y,2) (not-shifted)
    maps: (x,y,2*coils)
    """
    image = tf.expand_dims(image,0)
    maps = tf.expand_dims(maps,0)

    image = utils.join_reim_channels(image)
    image = tf.repeat(image, maps.shape[3], axis=3)
    sens_image = tf.cast(image,tf.complex64)*tf.cast(maps,tf.complex64)
    shift_sens_image = tf.signal.ifftshift(sens_image, axes=(1,2))
    shift_sens_image = utils.split_reim_channels(shift_sens_image)
 
    k = utils.convert_channels_to_frequency_domain(shift_sens_image)
    return k[0,...]

def rss_image_from_multicoil_k(k):
    """
    k: (x,y,2*coils) (shifted)
    """
    k = tf.expand_dims(k,0)
    img = utils.convert_channels_to_image_domain(k)
    img = tf.signal.fftshift(img, axes=(1,2))
    comp_img = utils.join_reim_channels(img)
    sq = K.square(K.abs(comp_img))
    return tf.expand_dims(K.sqrt(K.sum(sq,axis=3)),-1)[0,...]

def transform(sl, affine_mat):
    shape = tf.shape(sl)
    x = shape[0]
    y = shape[1]
    n = 600

    dx = tf.cast(n / 2 - x / 2, tf.int32)
    dy = tf.cast(n / 2 - y / 2, tf.int32)

    sl = tf.pad(sl, [[dx, dx], [dy, dy], [0,0]],mode='REFLECT')
    sl = tf.expand_dims(sl,0)
    sl.set_shape((None,n,n,2))
    sl_moved = vxm.layers.SpatialTransformer(interp_method='linear',fill_value=None)([sl,affine_mat])
    sl_moved = tf.slice(sl_moved, (0, dx, dy, 0), (-1, x, y, 2))[0,...]

    return sl_moved

def add_rotation_and_translations(sl, maps, order_ky, angle, num_pix, norms=1):
    """Add k-space rotations and translations to input slice.
    At each line in coord_list in k-space, induce a rotation and translation.
    Args:
      sl(float): Numpy array of shape (x, y, 2) containing input image data
      coord_list(int): Numpy array of (num_points) k-space line indices at which to induce motion
      angle(float): Numpy array of angles by which to rotate the input image; of shape (num_points)
      num_pix(float): List of horizontal and vertical translations by which to shift the input image; of shape (num_points, 2)
    Returns:
      sl_k_corrupt(float): Motion-corrupted k-space version of the input slice, of shape(n, n)
    """
    k_shape = tf.shape(sl)
    k_shape = [k_shape[0],k_shape[1],88]

    sl_k_combined = tf.cast(tf.zeros(k_shape),tf.float32)
    sl_k_true = tf.cast(tf.zeros(k_shape),tf.float32)

    for shot in tf.range(6):

        theta = angle[shot]
        theta = -theta*2*math.pi/360

        rot = tf.cast([[tf.cos(theta),-tf.sin(theta)],[tf.sin(theta),tf.cos(theta)]],tf.float32)
        rot = tf.expand_dims(rot,-1)
        rot = tf.transpose(rot, (2,0,1))

        v_trans = -tf.cast(num_pix[shot, 0], tf.float32)
        h_trans = -tf.cast(num_pix[shot, 1], tf.float32)

        trans = tf.expand_dims(tf.stack([h_trans,v_trans],axis=-1),-1)
        trans = tf.expand_dims(trans,0)

        affine_mat = tf.concat([rot,trans],axis=2)
        sl_moved = transform(sl,affine_mat)

        sl_k_after = simulate_multicoil_k(sl_moved,maps)

        if(tf.cast(order_ky[0,int(tf.shape(sl)[1]/2),0,shot],dtype=tf.bool)):
            sl_k_true = sl_k_after

        if(norms is not None):
            noise = tf.random.normal(shape=tf.shape(sl_k_after),stddev=10000)
            noise /= norms
            sl_k_after = sl_k_after + noise

        sl_k_combined += sl_k_after*order_ky[...,shot]

    return sl_k_combined, sl_k_true
