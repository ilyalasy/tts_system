import tensorflow as tf

from tensorflow_addons.seq2seq import AttentionWrapper, BahdanauAttention, BasicDecoder,dynamic_decode

import hyperparams as hp


class Tacotron(tf.keras.Model):
    def __init__(self,name='Tacotron'):
        super(Tacotron, self).__init__(name=name)
        # Embedding
        self.embedding = tf.keras.layers.Embedding(len(hp.vocab), hp.embed_size)

        # Encoder
        self.encoder = Encoder()
    
        # Attention Decoder RNN
        self.decoder_prenet = PreNet()
        self.attention_decoder = AttentionDecoder()
     
        # Post CBHG 
        self.post_cbhg = CBHG(256,80,out=True,k=8)

    def call(self, inputs, training=True, **kwargs): 
        mels = kwargs.get("mels", False)
        if not mels:
            raise ValueError("Expect to have mels param")

        encoder_input = self.embedding(inputs,training)

        decoder_inputs = tf.concat((tf.zeros_like(mels[:, :1, :]), mels[:, :-1, :]), 1) # (N, Ty/r, n_mels*r)
        decoder_inputs = decoder_inputs[:, :, -hp.n_mels:] # feed last frames only (N, Ty/r, n_mels)

        memory = self.encoder(encoder_input)
        memory = self.decoder_prenet(memory) 
        
        decoder_output = self.attention_decoder(inputs=decoder_inputs, memory=memory)

        return decoder_output, self.post_cbhg(decoder_output)

class AttentionDecoder(tf.keras.layers.Layer):
    def __init__(self):
        super(AttentionDecoder, self).__init__()

        self.gru1 = tf.keras.layers.GRUCell(hp.gru_size)
        self.gru2 = tf.keras.layers.GRUCell(hp.gru_size)

        self.dense_out = tf.keras.layers.Dense(hp.n_mels*hp.r)


    def call(self, inputs, training=True, **kwargs):
        memory = kwargs.get("memory", False)
        if not memory:
            raise ValueError("Expect to have memory param")

        attention_mechanism = BahdanauAttention(hp.gru_size, memory=memory)
        decoder_cell = tf.keras.layers.GRUCell(hp.gru_size)
        attention_cell = AttentionWrapper(decoder_cell, attention_mechanism, attention_layer_size=hp.gru_size, alignment_history=True)
        self.attention_rnn = tf.keras.layers.RNN(attention_cell,return_state=True)

        outputs, state = self.attention_rnn(inputs)

        ## for attention monitoring
        alignments = tf.transpose(state.alignment_history.stack(),[1,2,0])

        # Decoder RNNs
        outputs += self.gru1(outputs) # (N, T_y/r, E)
        outputs += self.gru2(outputs) # (N, T_y/r, E)

        # Outputs => (N, T_y/r, n_mels*r)
        mel_hats = self.dense_out(outputs)

        return mel_hats, alignments   

class Encoder(tf.keras.layers.Layer):
    def __init__(self):
        super(Encoder, self).__init__()
        self.pre_net = PreNet()
        self.cbhg = CBHG(128,128,out=False)

    def call(self, inputs,training):
        x = self.pre_net(inputs)
        return self.cbhg(x)


DENSE_SIZES = [256,128]
class PreNet(tf.keras.layers.Layer):
    def __init__(self):
        super(PreNet, self).__init__()
        self.fc1 = tf.keras.layers.Dense(DENSE_SIZES[0], activation=tf.nn.relu)
        self.dropout1= tf.keras.layers.Dropout(hp.dropout_rate)
        self.fc2 = tf.keras.layers.Dense(DENSE_SIZES[1], activation=tf.nn.relu)
        self.dropout2 = tf.keras.layers.Dropout(hp.dropout_rate) 

    def call(self, inputs,training):
        x = self.fc1(inputs)
        x = self.dropout1(x,training)
        x = self.fc2(x)
        return self.dropout2(x,training)

class CBHG(tf.keras.layers.Layer):
    def __init__(self,proj1,proj2,out,k=16):
        super(CBHG, self).__init__()
        self.convolutions = [tf.keras.layers.Conv1D(128,i,activation=tf.nn.relu, padding='same') for i in range(1,k+1)]
        self.batch_norms = [tf.keras.layers.BatchNormalization() for _ in range(k)]

        self.maxpool = tf.keras.layers.MaxPool1D(pool_size=2, strides=1,padding='same')

        self.conv_proj1 = tf.keras.layers.Conv1D(proj1, 3,activation=tf.nn.relu, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()

        self.conv_proj2 = tf.keras.layers.Conv1D(proj2,3, padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        
        self.dense = tf.keras.layers.Dense(hp.embed_size//2)

        self.highway = [HighwayNet() for _ in range(hp.num_highwaynet_blocks)]

        self.bidir_gru = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128))

        self.out_dense = tf.keras.layers.Dense(1+hp.n_fft//2)

        self.out = out


    def call(self, inputs, training):
        if self.out:
            # Restore shape -> (N, Ty, n_mels)
            inputs = tf.reshape(inputs, [tf.shape(inputs)[0], -1, hp.n_mels])

         # Convolution bank: concatenate on the last axis to stack channels from all convolutions
        conv_outputs = tf.concat([bn(conv(inputs),training) for conv, bn in zip(self.convolutions,self.batch_norms)], axis=-1)

        maxpool_output = self.maxpool(conv_outputs)
        proj1_output = self.bn1(self.conv_proj1(maxpool_output),training)
        output = self.bn2(self.conv_proj2(proj1_output),training)

        if self.out:
            output = self.dense(output)
        else:
            # Residual connection:
            highway_input = output + inputs

        # 4-layer HighwayNet:
        for i in range(hp.num_highwaynet_blocks):
            highway_input = self.highway[i](highway_input)

        rnn_input = highway_input
        outputs, _ = self.bidir_gru(rnn_input,training)

        if self.out:
            return self.out_dense(outputs)
        else:   
            # why outputs, not states?    
            return tf.concat(outputs, axis=2)  # Concat forward and backward

class HighwayNet(tf.keras.layers.Layer):
    def __init__(self):
        super(HighwayNet, self).__init__()
        self.h = tf.keras.layers.Dense(128, activation=tf.nn.relu)
        self.t = tf.keras.layers.Dense(128, activation=tf.nn.sigmoid,bias_initializer=tf.constant_initializer(-1.0))

    def call(self, inputs):
        h = self.h(inputs)
        t = self.t(inputs)
        return h * t + inputs * (1.0 - t)

