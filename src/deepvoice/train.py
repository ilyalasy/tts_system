from deepvoice import DeepVoice
import hyperparams as hp
import numpy as np
import tensorflow as tf


def loss(model, x, mel, mag,training):
    mel_pred, done_pred, mag_pred = model(x,mel,training)

    mae = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.SUM)

    loss_mel = mae(mae,mel,mel_pred)
    loss_mag = mae(mae,mel,mag_pred)            
    loss = loss_mel + loss_mag

    return loss_mel,loss_mag,loss

def grad(model, x, mel, mag):
  with tf.GradientTape() as tape:
    loss_mel,loss_mag,loss = loss(model, x,mel,mag, training=True)
  return loss_mel, loss_mag, loss, tape.gradient(loss, model.trainable_variables)

def train():
    model = DeepVoice()

    data =  np.array([[3,4,5],[1,2,3]])


    optimizer = tf.keras.optimizers.Adam(learning_rate=hp.lr,clipnorm=5.)
    
    # ce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    

    for i in hp.epochs:
        for x, mel, mag in data:
            loss_mel, loss_mag, loss, grads = grad(model,x,mel,mag)

            print(f"Step: {optimizer.iterations.numpy()}, Initial Loss: {loss_value.numpy()}")

            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            print(f"Step: {optimizer.iterations.numpy()}, Loss: {loss(model, x,mel,mag, training=True).numpy())}")


            



