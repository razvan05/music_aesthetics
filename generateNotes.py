import os, sys
from music21 import converter, instrument, note, chord, stream, midi, meter
import numpy as np
import random

import chromosome_transformer_fromServer as tr

import tensorflow as tf

class Config():
    model = None
      
def generate(generated_text_length):
    #define model hyperparameters
    config = Config()
    config.model = 'chromosome_transformer'
    config.learning_rate = 0.0001
    config.sequence_length = 50
    #config.hidden_values = 500
    config.embed_dim = 256
    config.num_layers = 2
    
    unique_notes = np.load('./models/unique_notes.npy', allow_pickle = True)
    unique_dur = np.load('./models/unique_dur.npy', allow_pickle = True)

    #integer encoding
    note_to_int = dict((notee, i) for i, notee in enumerate(unique_notes))
    dur_to_int = dict((dur, i) for i, dur in enumerate(unique_dur))
    
    int_to_note = dict((i, notee) for i, notee in enumerate(unique_notes))
    int_to_dur = dict((i, dur) for i, dur in enumerate(unique_dur))
    
    model = tr.transformer_model(config, len(note_to_int), len(dur_to_int))
    model.load_weights('./models/chromosome_transformer/my_checkpoint')
    
    
    # start_token = [tokenizer_pt.vocab_size]
    # end_token = [tokenizer_pt.vocab_size + 1]
    
    # # inp sentence is portuguese, hence adding the start and end token
    # inp_sentence = start_token + tokenizer_pt.encode(inp_sentence) + end_token
    # encoder_input = tf.expand_dims(inp_sentence, 0)
    
    # # as the target is english, the first word to the transformer should be the
    # # english start token.
    # decoder_input = [0]
    # output = tf.expand_dims(decoder_input, 0)
    input_note = tf.constant([[0] * 50]) #tf.random.uniform(shape = [1, 50], minval=0, maxval=len(note_to_int), dtype=tf.int32) #
    input_dur = tf.constant([[0] * 50]) #tf.random.uniform(shape = [1, 50], minval=0, maxval=len(dur_to_int), dtype=tf.int32) #tf.constant([[0] * 50])
    
    output_notes = []
    output_dur = []
    
    for i in range(generated_text_length):
      
        # predictions.shape == (batch_size, seq_len, vocab_size)
        predictions = model.predict_on_batch([input_note, input_dur])
        # print(len(predictions)) # pt input (2, 1) returneaza (2, 1, 50, vocab_size)
        
        # select the last note from the seq_len dimension
        notes_pred = predictions[0][: ,-1:, :]  
        # print(len(notes_pred[0][0]))  # (batch_size, 1, notes_vocab_size)
        predicted_note_id = tf.cast(tf.argmax(notes_pred, axis = -1), tf.int32)
        input_note = tf.concat([input_note[:, 1:], predicted_note_id], axis = -1)
        output_notes.append(tf.squeeze(predicted_note_id, axis = 0).numpy().tolist()[0]) # elimin dimensiunea utilizata pt batch_nr si parsez la numpy
        
        dur_pred = predictions[1][: ,-1:, :]  
        # print(len(dur_pred[0][0]))  # (batch_size, 1, dur_vocab_size)
        predicted_dur_id = tf.cast(tf.argmax(dur_pred, axis = -1), tf.int32)
        input_dur = tf.concat([input_dur[:, 1:], predicted_dur_id], axis = -1)
        output_dur.append(tf.squeeze(predicted_dur_id, axis = 0).numpy().tolist()[0]) # elimin dimensiunea utilizata pt batch_nr si parsez la numpy

     
    # decodez din int in note/dur
    output_notes = [int_to_note[notee] for i, notee in enumerate(output_notes)]
    output_dur = [int_to_dur[dur] for i, dur in enumerate(output_dur)]
    
    print(output_notes)
    print(output_dur)
    
    return output_notes, output_dur

output_notes, output_dur = generate(250)
output_notes = output_notes[50:]
output_dur = output_dur[50:]


# generate melody
m = meter.TimeSignature('2/4')
s = stream.Score(id='mainScore')
p = stream.Part(id = 'part' + str(1))  # vocea 1
p.append(m)

for i, n in enumerate(output_notes):
    nn = note.Note(str(n))
    nn.quarterLength = float(output_dur[i])
    p.append(nn)
    
s.insert(1, p)
s.show('text')
m = midi.translate.streamToMidiFile(s)
m.open('melodie.mid', 'wb')
m.write()
m.close()