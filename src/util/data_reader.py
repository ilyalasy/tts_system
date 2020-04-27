import pandas as pd
import numpy as np

import hyperparams as hp
import util.audio as audio
import util.text_util as text_util
import tensorflow as tf


class DataReader():
    def __init__(self, data_path=hp.data_path):
        self.wavs_path = data_path
        self.labels_path = data_path + hp.transcript_path

        self._df = pd.read_csv(self.labels_path, sep='|', header=None, names=['wav_path','text', 'text_norm','duration'])    
        self._normalize_text()
        self.df = self._df[self._df['duration'] < hp.max_duration].copy()
        self.df.drop(['text_norm'], axis=1, inplace=True)

    def _normalize_text(self):
        self._df['text'] = self._df['text'].apply(text_util.text_normalize)
    
    @property
    def max_duration(self):
        return self.df['duration'].max()

    @property
    def max_characters(self):
        return len(self.df['text'].max())

    @property
    def total_audio_len(self):
        return self.df['duration'].sum()


    def _process_sample(self, row, load=False):
        
        if load:
            pass
        else:
            wav = audio.load_wav(self.wavs_path + row['wav_path'])

            # Compute the linear-scale spectrogram from the wav:
            spectrogram = audio.spectrogram(wav).astype(np.float32).T
            # n_frames = spectrogram.shape[1]

            # Compute a mel-scale spectrogram from the wav:
            mel_spectrogram = audio.melspectrogram(wav).astype(np.float32).T

        dones = np.zeros(mel_spectrogram.shape[0])
        dones[-1] = 1

        char2idx, _ = text_util.get_vocab()
        text = [char2idx[char] for char in row['text']]

        frames_count = mel_spectrogram.shape[0]
        # Padding
        text = tf.pad(text, ((0, hp.max_timesteps),))[:hp.max_timesteps] # (max_timesteps,)
        mel_spectrogram = tf.pad(mel_spectrogram, ((0, hp.max_frames), (0, 0)))[:hp.max_frames] # (max_frames, n_mels)
        dones = tf.pad(dones, ((0, hp.max_frames),))[:hp.max_frames] # (max_frames,)
        spectrogram = tf.pad(spectrogram, ((0, hp.max_frames), (0, 0)))[:hp.max_frames] # (max_frames, 1+n_fft/2)

        return text, spectrogram, mel_spectrogram, dones, frames_count

    def get_data(self,n=hp.batch_size):
        batch_x,batch_mag,batch_mel,batch_dones,batch_frames = [], [], [], [], []
        for _, row in self.df.iterrows():
            
            text, mag, mel, dones,size = self._process_sample(row)

            batch_x.append(text)
            batch_mag.append(mag)
            batch_mel.append(mel)
            batch_dones.append(dones)
            batch_frames.append(size)

            if len(batch_x) == n:

                # x = self._list_to_ragged(batch_x)
                # mag = self._list_to_ragged(batch_mag)
                # mel = self._list_to_ragged(batch_mel)
                # dones = self._list_to_ragged(batch_dones)
                
                yield tf.convert_to_tensor(batch_x), \
                    tf.convert_to_tensor(batch_mag), \
                    tf.convert_to_tensor(batch_mel), \
                    tf.convert_to_tensor(batch_dones), \
                    tf.convert_to_tensor(batch_frames)

                batch_x,batch_mag,batch_mel,batch_dones = [], [], [], []

    def _list_to_ragged(self, values):
        lengths = [len(v) for v in values]
        flatten = np.concatenate(values, axis=0)
        return tf.RaggedTensor.from_row_lengths(values=flatten,row_lengths=lengths)

# next(DataReader().get_data())