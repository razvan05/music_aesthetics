import sys
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LayerNormalization
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import GlobalAveragePooling1D
from tensorflow.keras import Model
from tensorflow.keras import Sequential
import tensorflow.keras.backend as K
import numpy as np

# epic
# https://stats.stackexchange.com/questions/421935/what-exactly-are-keys-queries-and-values-in-attention-mechanisms

# querry = output decoder = ceea ce vrem (S0, S1 din scapple)
# (target sequence la traducere, recomandari youtube)

# key = h din scapple = input data (pt layerul din mijlocul decoderului, inputul este outputul de la encoder)
# key = values in unele situatii
# (source sequence la traducere, din user profile and history)

# values = c din scapple = input data
# (source sequence la traducere, din user profile and history)
    
def scaled_dot_product_attention(keys, queries, values, mask):
    score = tf.matmul(queries, keys, transpose_b = True) / tf.sqrt(tf.cast(tf.shape(keys)[-1], tf.float32))
    
    if mask is not None:
        # score += (mask * -1e9) 
        score = ((score * mask) + (1.0 - mask)) * (-1e10)
    
    attention = tf.nn.softmax(score, axis = -1)
    output = tf.matmul(attention, values)
    return output

def split_heads(x, batch_size, num_heads, head_dim):
    """Split the last dimension into (num_heads, head_dim).
    Transpose the result such that the shape is (batch_size, num_heads, seq_len, head_dim)
    """
    x = tf.reshape(x, (batch_size, -1, num_heads, head_dim))
    return tf.transpose(x, perm=[0, 2, 1, 3])

def multi_head_attention(keys, queries, values, embed_dim, num_heads, mask):
    assert embed_dim % num_heads == 0
    head_dim = embed_dim // num_heads
    
    batch_size = tf.shape(queries)[0]
    # key and queries are assumed to have the same dimensionality
    qD = Dense(embed_dim)
    queries = qD(queries)
    queries = split_heads(queries, batch_size, num_heads, head_dim)
    
    kD = Dense(embed_dim)
    keys = qD(keys)
    keys = split_heads(keys, batch_size, num_heads, head_dim)
        
    vD = Dense(embed_dim)
    values = qD(values)
    values = split_heads(values, batch_size, num_heads, head_dim)
    #print(values)
    attention = scaled_dot_product_attention(keys, queries, values, mask)
    attention = tf.transpose(attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

    attention = tf.reshape(attention, (batch_size, -1, embed_dim))  # (batch_size, seq_len_q, d_model)
    #print(head) # encoder 8 x (None, 50, 32), decoder = (None, 50, 32)
    out = Dense(embed_dim, activation=None, use_bias=False)
    #out.build(input_shape = (tf.shape(queries)[0], tf.shape(queries)[1], num_heads * head_dim))
    output = out(attention)
    #print(output) # encoder (None, 50, 256) decoder (None, 50, 256)
    return output

def transformer_block(keys, queries, values, embed_dim, mask = None):
    #dim = tf.cast(tf.shape(keys)[-1], 'float32')
    attention_out = multi_head_attention(keys, queries, values, embed_dim, 8, mask)
    #print(attention_out) # encoder (None, 50, 256) decoder (None, 50, 256)
    
    dropout1 = Dropout(0.1)
    layer_norm_1 = LayerNormalization(axis = -1, epsilon = 1e-6)
    x = layer_norm_1(dropout1(attention_out + queries, training = True))
    #print(x) # encoder (None, 50, 256) decoder (None, 50, 256)
    
    feed_forward_network = Sequential(
            [Dense(embed_dim, activation="relu"), Dense(embed_dim),]
        )
    feed_forward_out = feed_forward_network(x)
    #print(feed_forward_out) # encoder (None, 50, 256) decoder (None, 50, 256)
    dropout2 = Dropout(0.1)
    layer_norm_2 = LayerNormalization(axis = -1, epsilon = 1e-6)
    out = layer_norm_2(dropout2(feed_forward_out + x, training = True))
    return out

def get_angles(pos, i, embed_dim):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(embed_dim))
    return pos * angle_rates

def positional_encoding(seq_len, embed_dim):
    angle_rads = get_angles(np.arange(seq_len)[:, np.newaxis], np.arange(embed_dim)[np.newaxis, :], embed_dim)
    
    # apply sin to even indices in the array; 2i
    angle_rads[0::2] = np.sin(angle_rads[0::2])
    
    # apply cos to odd indices in the array; 2i+1
    angle_rads[1::2] = np.cos(angle_rads[1::2])
      
    pos_encoding = angle_rads[np.newaxis, ...]
      
    return tf.cast(pos_encoding, dtype=tf.float32)

# x = (None, 50) encoder, embed_dim = 256
def token_and_position_embedding(x, embed_dim, seq_len, vocab_size):
    # maxlen = tf.shape(x)[-1]  # x este un Input empty la decraratie (are valoare la call decat)
    # tf.print(maxlen, output_stream = sys.stdout)

    # positions = tf.keras.backend.arange(start = 0, stop = seq_len, step = 1)
    # pos_emb = Embedding(input_dim = seq_len, output_dim = embed_dim, input_length = seq_len)
    # positions = pos_emb(positions)
    # #print(positions) # encoder = (50, 256), decoder = (50, 256)
    # positions = tf.expand_dims(positions, axis = 0)
    # #print(positions) # encoder = (1, 50, 256), decoder = (1, 50, 256)
    positions = positional_encoding(seq_len, embed_dim)
    
    token_emb = Embedding(input_dim = vocab_size, output_dim = embed_dim, input_length = seq_len)
    x = token_emb(x) # inputurile din encoder/decoder sunt deja hot encoded (nu au nevoie de embedding) !!!!!!
    # dense = Dense(embed_dim)
    # x = dense(x)

    x *= tf.math.sqrt(tf.cast(embed_dim, tf.float32))
    #print(x) # encoder = (None, 50, 256), decoder = (None, 50, 256)
    out = x + positions[:, :seq_len, :] # adun pozitiile pentru fiecare secventa din batch in parte
    return out

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)

# encoder_in = (None, 50)
def encoder(encoder_in, num_layers, embed_dim, seq_len, vocab_size):
    encoder_embedded_input = token_and_position_embedding(encoder_in, embed_dim, seq_len, vocab_size) 
    #print(encoder_embedded_input) (None, 50, 256)
    dropout = Dropout(0.1)
    l = LayerNormalization(epsilon = 1e-6)
    x = l(dropout(encoder_embedded_input, training = True)) #(None, 50, 256)
    
    #mask = Lambda(create_look_ahead_mask)(seq_len) #(50, 50) cu 1 deasupra diag principala
    mask = create_look_ahead_mask(seq_len)
    # tf.print(enc_out, output_stream = sys.stdout)

    for i in range(num_layers):
        x = transformer_block(x, x, x, embed_dim, mask)
    output = x
    #print(attention) #(None, 50, 256)
    
    return output

#embedd_dim = 256, seq_len = 50, enc_out = (None, 50, 256)
def decoder(same_enc_out, other_enc_out, num_layers, embed_dim, seq_len, vocab_size):
    x = transformer_block(other_enc_out, same_enc_out, other_enc_out, embed_dim) #(None, 50, 256)
    for i in range(num_layers):
        x = transformer_block(x, x, x, embed_dim)
        
    #out = GlobalAveragePooling1D(data_format='channels_first')  # scapi de embedding (la final ai sir de 50 note in locul (50, 256))

    out = Dense(vocab_size)
    out = out(x)
    #print(out) #(None, 50)

    #out = tf.squeeze(dense) 
    #print(out)
    return out

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
    
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        
        self.warmup_steps = warmup_steps
    
    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
    
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        config = {
        'd_model': self.d_model,
        'warmup_steps': self.warmup_steps,
         }
        return config
    
def transformer_model(config, notes_encoding_size, durations_encoding_size):
    seq_len = config.sequence_length
    embed_dim = config.embed_dim
    num_layers = config.num_layers
    
    notes_encoder_input = Input(shape = (seq_len, )) 
    #print(source_encoder_input) # (None, 50)
    encoder1_out = encoder(notes_encoder_input, num_layers, embed_dim, seq_len, notes_encoding_size) 
    #print(encoder1_out) # (None, 50, 256)

    dur_decoder_input = Input(shape = (seq_len, ))
    encoder2_out = encoder(dur_decoder_input, num_layers, embed_dim, seq_len, durations_encoding_size) 
    #print(encoder2_out) # (None, 50, 256)
    
    decoder1_out = decoder(encoder1_out, encoder2_out, num_layers, embed_dim, seq_len, notes_encoding_size) #(None, 50)
    decoder2_out = decoder(encoder2_out, encoder1_out, num_layers, embed_dim, seq_len, durations_encoding_size) #(None, 50)
    
    # decoder_out = Lambda(decoder)([target_decoder_input, encoder_out, notes_encoding_size, embed_dim, seq_len])
    model = Model([notes_encoder_input, dur_decoder_input], [decoder1_out, decoder2_out])

    learning_rate = CustomSchedule(embed_dim)
    opt = tf.keras.optimizers.Adam(learning_rate)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True, reduction = 'none')
    model.compile(loss = loss, optimizer = opt, metrics = ['accuracy'], run_eagerly = True)

    return model
    
    
    
    
