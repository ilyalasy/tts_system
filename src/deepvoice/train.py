from deepvoice import DeepVoice
import hyperparams as hp
import numpy as np
import tensorflow as tf
from util.data_reader import DataReader
import datetime

mae = tf.keras.losses.MeanAbsoluteError(
      reduction=tf.keras.losses.Reduction.SUM)
ce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

def masked_l1(y_true,y_pred,mask):
  return mae(y_true,y_pred,mask)

def masked_crossentropy(y_true,y_pred,mask):
  return ce(y_true,y_pred,mask)
  
def create_mask(current_shape, orig_shapes):
  mask = np.ones(current_shape[:2])
  for b in range(current_shape[0]):
    mask[b,orig_shapes[b]:-1] = 0
  return tf.convert_to_tensor(mask)


def get_loss(model, x, mel, done, mag, frames, training):

    mel_pred, done_pred, mag_pred = model(x, mel, training)

    mel_mask = create_mask(mel.shape,frames)
    mag_mask = create_mask(mag.shape,frames)
    
    done_mask = create_mask(done.shape,frames)

    loss_mel = masked_l1(mel, mel_pred,mel_mask)
    loss_mag = masked_l1(mag, mag_pred,mag_mask)
    loss_done = masked_crossentropy(done, done_pred,done_mask)
    loss = loss_mel + loss_done + loss_mag

    return loss_mel, loss_done, loss_mag, loss


def grad(model, x, mel, done, mag, sizes):
  with tf.GradientTape() as tape:
    loss_mel, loss_done, loss_mag, loss = get_loss(
        model, x, mel, done, mag, sizes, training=True)
  return loss_mel, loss_done, loss_mag, loss, tape.gradient(loss, model.trainable_variables)


def train():
    # init model
    model = DeepVoice()
    print(f'Model {model.name} created!')

    # get data
    reader = DataReader()
    data = reader.get_data()
    print(f'Initialized data with total audio length: {reader.total_audio_len / 3600} hours')

    # setup optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=hp.lr, clipnorm=5.)

    # setup writer
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = 'logs/' + current_time + '/'
    checkpoints_dir = 'checkpoints/'
    summary_writer = tf.summary.create_file_writer(log_dir)

    print(f'Logs in {log_dir}.')
    print(f'Checkpoints in {checkpoints_dir}.')
    print(f'Training started for {hp.epochs} epochs.')
    for epoch in range(hp.epochs):
      print(f'### Epoch {epoch} ###')
      for x, mag, mel, done, frame_sizes in data:
        done = done[:,::hp.r]
        
        loss_mel, loss_done, loss_mag, loss, grads = grad(
            model, x, mel, done, mag,frame_sizes)

        print(f"Step: {optimizer.iterations.numpy()}, Loss: {loss.numpy()}")

        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        with summary_writer.as_default():
          tf.summary.scalar('Loss', loss, step=optimizer.iterations.numpy())
          tf.summary.scalar('mels', loss_mel, step=optimizer.iterations.numpy())
          tf.summary.scalar('dones', loss_done, step=optimizer.iterations.numpy())
          tf.summary.scalar('mags', loss_mag, step=optimizer.iterations.numpy())
      
      mae.reset_states()
      ce.reset_states()
      
      # if epoch % 2 == 0:
      time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
      checkpoints_path = checkpoints_dir + time + ''
      model.save_weights(checkpoints_path)


if __name__ == "__main__":
    train()
