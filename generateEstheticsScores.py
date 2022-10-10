import os, sys
from music21 import converter, instrument, note, chord, stream, midi, meter, interval
import numpy as np

import bilstm

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Model
from tqdm import tqdm
#from random import shuffle
import gc

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#     try:
#         tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2000)])
#     except RuntimeError as e:
#         print(e)
class Config():
    model = None
    
def generate():       
    notes = []
    durations = []
    
    notes_not_disonances = []
    durations_not_disonances = []

    #define model hyperparameters
    config = Config()
    config.model = 'bigru'
    config.learning_rate = 0.0001
    config.sequence_length = 20
    config.hidden_values = 200
    config.epochs = 1
    config.batch_size = 32
    
    unique_notes = np.load('./models/unique_notes.npy', allow_pickle = True)
    unique_dur = np.load('./models/unique_dur.npy', allow_pickle = True)

    #integer encoding
    note_to_int = dict((note, i) for i, note in enumerate(unique_notes))
    dur_to_int = dict((dur, i) for i, dur in enumerate(unique_dur))
    
    encoded_notes = [note_to_int[note] for note in unique_notes]    #126 (18 game * 7 note)
    encoded_dur = [dur_to_int[dur] for dur in unique_dur]           #5159
    model, _, _ = bilstm.near_disonances_notes_model(config, len(to_categorical(encoded_notes)), len(to_categorical(encoded_dur)))
    model.load_weights('./models/bigru/my_checkpoint')
    
    
    #read melody
    midi_file = converter.parse('melodie' + str(5) + '.mid')
    
    m = meter.TimeSignature('4/4')
    s = stream.Score(id='mainScore')

    voice = stream.Part(id = 'part' + str(1))
    voice.append(m)
    notes_to_parse = midi_file.flat.notes
    
    notes_seq = []
    dur_seq = []
    
    good_disonances = 0
    bad_disonances = 0
    
    for element in notes_to_parse: # pt fiecare nota din melodia curenta
        if isinstance(element, note.Note):
            current_note = element.pitch
            notes_seq.append(current_note)
            
            current_dur = element.duration.quarterLength
            dur_seq.append(current_dur)
            
            s_len = config.sequence_length
            
            if len(notes_seq) == s_len:  # seq of 20 notes
            
                # avem o disonanta reala
                if interval.Interval(noteStart = notes_seq[s_len // 2 - 2], noteEnd = notes_seq[s_len // 2 - 1]).simpleName in ["m2", "M2", "TT", "m7", "M7"]: 
                    
                    #encode notes/dur for NN input
                    encoded_notes = [note_to_int[str(note)] for note in notes_seq]    #126 (18 game * 7 note)
                    encoded_dur = [dur_to_int.get(str(dur), 0) for dur in dur_seq]
                    
                    encoded_notes = to_categorical(encoded_notes, num_classes = len(note_to_int))   
                    encoded_dur = to_categorical(encoded_dur, num_classes = len(dur_to_int))
                    
                    #predict if there is a disonance between note 9 and 10
                    prediction = model.predict_on_batch([tf.expand_dims(encoded_dur, axis = 0), 
                                                         tf.expand_dims(encoded_notes, axis = 0)])
                    #print(prediction)
                    #prediction = tf.math.reduce_sum(prediction, axis = 1)
                    prediction = tf.squeeze(prediction, axis = 0)
                    #print(prediction)
                
                    if prediction[0] < 0.25: # prezice ca nu este disonanta
                        bad_disonances += 1
                    else: # prezice ca este disonanta
                        good_disonances += 1
                        
                notes_seq.pop(0)
                dur_seq.pop(0)
                
    if good_disonances + bad_disonances == 0:
        print("nu exista disonante")
    else:
        print("Good disonances:  " + str(good_disonances))
        print("Bad disonances:  " + str(bad_disonances))

generate()
