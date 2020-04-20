import pandas as pd
import numpy as np

import hyperparams as hp
import util.audio as audio
import util.text_util as text_util


import tensorflow as tf
class DataReader():
    def __init__(self, data_path):
        self.data_path = data_path + hp.data
        self.labels_path = data_path + hp.data_csv

        self._df = pd.read_csv(self.labels_path, names=['wav_path','text_path','duration'])
        self._df['wav_path'] = self._df['wav_path'].str.replace('../data/ru_open_stt/russian_single/', '')
        self._df['text_path'] = self._df['text_path'].str.replace('../data/ru_open_stt/russian_single/', '')
        

    @property
    def df(self):
        return self._df[self._df['duration'] < hp.max_duration]

    @property
    def total_audio_len(self):
        return self.df['duration'].sum()

    def get_data(self,n=32):
        char2idx, idx2char = text_util.get_vocab()

        x, y, z = [], [], []
        batch_x, batch_y, batch_z = [], [], []
        for i, row in self.df.iterrows():
            mel, mag = audio.load_spectrograms(self.data_path + row['wav_path'])
            text = open(self.data_path + row['text_path'], encoding='utf8').read().strip() + 'Е'
            text = np.array([char2idx[char] for char in text], dtype=np.int32)

            x.append(text)
            y.append(mel)
            z.append(mag)
            if (i+1) == n:
                break

            # batch_x.append(text)
            # batch_y.append(mel)
            # batch_z.append(mag)
            # if len(batch_x) == hp.batch_size:
            #     if generator:
            #         yield np.array(batch_x,dtype=np.int32),np.array(batch_y, dtype=np.float32),np.array(batch_z, dtype=np.float32)
            #     else:
            #         x.append(np.asarray(batch_x))
            #         y.append(np.asarray(batch_y))
            #         z.append(np.asarray(batch_z)) 
            #     batch_x, batch_y,batch_z = [], [], []

        # if not generator:
        return np.asarray(x),np.asarray(y),np.asarray(z)

    # def get_data(self):
    #     char2idx, idx2char = text_util.get_vocab()

    #     x, y = [], []
    #     for _, row in self.df.iterrows():
    #         mel, mag = audio.load_spectrograms(self.data_path + row['wav_path'])
    #         text = open(self.data_path + row['text_path'], encoding='utf8').read().trim() + 'Е'
    #         text = [char2idx[char] for char in text]
    #         x.append(text)
    #         y.append( (mel,mag) )
    #     return np.asarray(x), np.asarray(y)


    # def clear(self):
    #     try:
    #         shutil.rmtree(self.path_to_delete)
    #         print ('Extracted files succesfully deleted!')
    #     except OSError as e:
    #         print (f'Error: {e.filename} - {e.strerror}.')

    # def extract_random_samples(self, n=1):
    #     sampled_df = self.df[['wav_path','text_path']].sample(n=n, random_state=42)
        
    #     with tarfile.open(self.tar_path, "r:gz") as tar:
    #         members = [m for m in tar.getmembers() if m.isreg() and m.name.replace('./','') in ''.join(sampled_df.values.flatten())]
    #         tar.extractall(self.base_data, members=members)
    #         print(f'Successfully extracted: {members}')

    #     samples = []
    #     for values in sampled_df.values:
    #         wav = wavfile.read(values[0])
    #         txt = open(values[1], encoding='utf8').read()
    #         samples.append((wav,txt))

    #     return samples  

dr = DataReader('D:/Dev/ML/TTS/data')

x,y,z = dr.get_data()

dataset1 = tf.data.Dataset.from_generator(lambda: x, 
                                         tf.as_dtype(x[0].dtype))


dataset2 = tf.data.Dataset.from_tensor_slices([y,z])

print('lol')
