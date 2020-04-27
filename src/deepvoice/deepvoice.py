import tensorflow as tf

from tensorflow.keras import layers
from tensorflow_addons.seq2seq import AttentionWrapper, BahdanauAttention, BasicDecoder,dynamic_decode

import hyperparams as hp

import numpy as np

class DeepVoice(tf.keras.Model):
    def __init__(self,name='DeepVoice'):
        super(DeepVoice, self).__init__(name=name)
    
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.converter = Converter()
    
    def call(self, inputs, mels, training=True): 
        keys, values = self.encoder(inputs,training)

        # Reduction
        mels = tf.reshape(mels, (-1,hp.max_frames//hp.r,hp.n_mels * hp.r))
        mels = tf.concat((tf.zeros_like(mels[:, :1, -hp.n_mels:]), mels[:, :-1, -hp.n_mels:]), axis=1) # Take evry 
        
        mel_output, done, dec_output = self.decoder(mels, keys, values,training)
        
        mag_output = self.converter(dec_output, training)

        mel_output = tf.reshape(mel_output, (-1,hp.max_frames,mel_output.shape[2]//hp.r))
        mag_output = tf.reshape(mag_output, (-1,hp.max_frames,mag_output.shape[2]//hp.r))
        
        return mel_output, done, mag_output
        

class Encoder(layers.Layer):
    def __init__(self,name='Encoder'):
        super(Encoder, self).__init__(name=name)
         # Embedding
        self.embedding = layers.Embedding(len(hp.vocab), hp.embed_size)
       
        self.fc1 = layers.Dense(hp.num_channels)

        self.convs = [ConvBlock(hp.num_channels * 2, dilation=2**i) for i in range(hp.encoder_layers)]

        self.fc2 = layers.Dense(hp.embed_size)
    
    def call(self, inputs,training):
        embeddings = self.embedding(inputs)
        conv_input = self.fc1(embeddings)
        for conv in self.convs:
            conv_input = conv(conv_input,training)
        keys = self.fc2(conv_input)
        values = (embeddings + keys) * tf.sqrt(0.5)
        return keys, values

class Decoder(layers.Layer):
    def __init__(self,name='Decoder'):
        super(Decoder, self).__init__(name=name)

        self.first_fc = layers.Dense(hp.embed_size,activation='relu')
        self.dropouts = [layers.Dropout(hp.dropout) for _ in range(hp.decoder_layers - 1)]
        self.fcs = [layers.Dense(hp.embed_size,activation='relu') for _ in range(hp.decoder_layers - 1)]

        self.convs = [ConvBlock(hp.embed_size * 2,padding='causal', dilation=2**i) for i in range(hp.decoder_layers)]
        self.attentions = [AttentionBlock(num_channels=hp.embed_size, query_pr=1.,key_pr=(hp.max_frames // hp.r) / hp.max_timesteps) for _ in range(hp.decoder_layers)]
        self.fc_done = layers.Dense(2,activation='sigmoid')
        self.fc_mel = layers.Dense(hp.n_mels * hp.r)

    def call(self, inputs, keys, values, training=True):
        fc_outputs = self.first_fc(inputs)
        for dropout, fc in zip(self.dropouts,self.fcs):
            fc_outputs = fc(dropout(fc_outputs,training))
        
        conv_output = fc_outputs
        
        for conv, attention in zip(self.convs, self.attentions):
            conv_output = conv(conv_output,training)
            att_output = attention(conv_output, keys, values,training)
            conv_output = (att_output + conv_output) * tf.sqrt(0.5)

        dec_output = conv_output

        done_output = self.fc_done(dec_output)


        mel_output = self.fc_mel(dec_output)

        return mel_output, done_output, dec_output

class AttentionBlock(layers.Layer):
    def __init__(self, num_channels, query_pr,key_pr,name='AttentionBlock'):
        super(AttentionBlock, self).__init__(name=name)
        self.num_channels = num_channels
        self.key_pr = key_pr
        self.query_pr = query_pr

        self.fc1 = layers.Dense(hp.attention_size)
        self.fc2 = layers.Dense(hp.attention_size)
        self.fc3 = layers.Dense(hp.attention_size)

        self.dropout = layers.Dropout(hp.dropout)

        self.fc_out = layers.Dense(hp.embed_size)


    def _get_angles(self, pos, pos_rate, k):
        angle_rates = pos * pos_rate / np.power(10000, 2 * (k//2) / self.num_channels)
        return pos * angle_rates

    def _get_pe(self, inputs,pos_rate):
        timesteps = inputs.shape.as_list()[1]

        position_enc = np.array([
            [self._get_angles(pos, pos_rate,  i) for i in range(self.num_channels)]for pos in range(timesteps)])
        
        # apply sin to even indices in the array; 2i
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])
  
        # apply cos to odd indices in the array; 2i+1
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])

        position_enc = position_enc[np.newaxis, ...]

        return tf.cast(position_enc, dtype=tf.float32)

    def call(self, query, keys, values, training=True):
        pe_query = self._get_pe(query,self.query_pr)
        pe_keys = self._get_pe(keys,self.key_pr)

        query = pe_query + query
        keys = pe_keys + keys

        keys_out = self.fc1(keys)
        query_out = self.fc2(query)
        values_out =  self.fc3(values)

        attention_weights = tf.matmul(query_out, keys_out, transpose_b=True)

        #key mask
        key_masks = tf.ones_like(tf.math.reduce_sum(keys_out, axis=-1))  # (N, Tx)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(query_out)[1], 1])  # (N, Ty/r, Tx)

        paddings = tf.ones_like(attention_weights) * (-2 ** 32 + 1)
        attention_weights = tf.where(tf.equal(key_masks, 0), paddings, attention_weights)  # (N, Ty/r, Tx)


        # softmax or monotonic
        alignments = tf.nn.softmax(attention_weights)

        # dropout
        alignments = self.dropout(alignments,training)

        # context 
        values_ts = values.shape.as_list()[1]
        context = tf.matmul(alignments, values_out)
        context *= tf.math.rsqrt(tf.cast(values_ts, dtype=tf.float32))

        return self.fc_out(context)

class ConvBlock(layers.Layer):
    def __init__(self, output_size, padding='same', dilation=1,name='ConvBlock'):
        super(ConvBlock, self).__init__(name=name)
        self.dropout= layers.Dropout(hp.dropout)
        self.conv = layers.Conv1D(output_size, hp.kernel_size,
                                    padding=padding,
                                    dilation_rate=dilation,
                                    kernel_initializer='glorot_normal')
        self.glu = GLU()

    def call(self, inputs,training):
        conv_input = self.dropout(inputs,training)
        glu_input = self.conv(conv_input)
        outputs = self.glu(glu_input)
        return (inputs + outputs ) * tf.sqrt(0.5)
        
class GLU(layers.Layer):
    def __init__(self,name='GLU'):
        super(GLU, self).__init__(name=name)

    def call(self, inputs):   
        a, b = tf.split(inputs, 2, -1)  # (N, Tx, c) * 2
        outputs = a * tf.math.sigmoid(b) # a + speaker embedding
        return outputs

class Converter(layers.Layer):
    def __init__(self, vocoder_type='griffin-lim', name='Converter'):
        super(Converter, self).__init__(name=name)

        self.convs = [ConvBlock(hp.embed_size * 2,dilation=2**i) for i in range(hp.converter_layers)]
        self.fc_out = layers.Dense(hp.converter_channels)

        # TODO: WORLD
        assert vocoder_type == 'griffin-lim'
        self.fc_gl = layers.Dense((hp.n_fft // 2 + 1) * hp.r)

    def call(self, inputs,training):
        conv_input = inputs
        for conv in self.convs:
            conv_input = conv(conv_input,training)

        out = self.fc_out(conv_input)
        mag_out = self.fc_gl(out) 
        return mag_out


    