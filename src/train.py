import tensorflow as tf
    
from tacotron.tacotron import Tacotron
from util.data_reader import DataReader

import hyperparams as hp

import datetime


def loss(model, x, y, z, training):

    y_pred, z_pred = model(x,mels=y,training=training)
    loss_mel = tf.reduce_mean(tf.abs(y_pred - y))
    loss_lin = tf.reduce_mean(tf.abs(z_pred - z))
    return loss_mel + loss_lin

def grad(model, inputs, mel, mag):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, mel, mag, training=True)
  return loss_value, tape.gradient(loss_value, model.trainable_variables)

def train():
    data_path = 'D:/Dev/ML/TTS/data'

    model = Tacotron()

    dr = DataReader(data_path)
    print(f'Ready to use dataset at {dr.data_path}.')
    print(f'{len(dr.df)} audio files with total duration - {dr.total_audio_len / 3600} hours.')

    dataset = dr.get_batches(generator=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=hp.lr,clipnorm=5.)


    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    
    num_epochs = 200
    print(f'Started training for {num_epochs} epochs.')
    for epoch in range(num_epochs):
        # Training loop - using batches of hp.batch_size
        for x, y, z in dataset:
            # Optimize the model
            loss_value, grads = grad(model, x, y,z)

            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            with train_summary_writer.as_default():
                # Summary
                tf.summary.scalar('loss', loss_value,step=epoch)
        
        template = 'Epoch {}, Loss: {}'
        # End epoch
        # train_loss_results.append(loss_value)

        print (template.format(epoch+1,loss_value))

if __name__ == '__main__':
    train()