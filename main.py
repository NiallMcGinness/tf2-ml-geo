from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np
import os
from tensorflow.keras import layers
from tensorflow.keras.initializers import he_normal
import time

from data_gen_utils import  create_data
from model_utils import create_model

# enable automatic 16 bit floating point  precisionm , automaticu mixed floating point  ( AMP ) 
# 
policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
tf.keras.mixed_precision.experimental.set_policy(policy)

dataset_size = 10000
image_size = 64
images, labels = create_data( dataset_size, image_size) 

images = tf.cast(images, tf.float32) / 255
labels = tf.cast(labels, tf.float32) / image_size

batch_size = 256
train_dataset = tf.data.Dataset.from_tensor_slices(  ( images, labels ) ).shuffle(dataset_size).batch(batch_size)


model = create_model()
MSE = tf.keras.losses.MSE
optimizer = tf.keras.optimizers.Adam(1e-4)

# adding tf.function decorator lays out the graph statically
# rather than the default eager mode, the static graph is faster to compute at runtime 
@tf.function
def train_step(images,labels):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = MSE(predictions, labels)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)


def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()
        step_loss = 0
        for batch in dataset:
            img_batch, label_batch = batch
            step_loss = train_step(img_batch, label_batch)
        if (epoch + 1) % 5 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)
            print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
            print('loss : {} '.format(tf.reduce_sum(step_loss) ) )




epochs = 100

train(train_dataset, epochs)
