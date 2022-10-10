import os, sys
from music21 import converter, instrument, note, chord
import numpy as np

import bilstm

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Model
from tqdm import tqdm
#from random import shuffle
import wandb
import gc

# wandb.login()
# wandb.init(project='neuralEsthetics', entity='razvan')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#     try:
#         tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=7000)])
#     except RuntimeError as e:
#         print(e)
    
class DurationsToPitchs:

    #load data from preprocessed files
    def start(self):        
        notes = []
        durations = []
        
        notes_not_disonances = []
        durations_not_disonances = []

        i = 0
        files_batch = 0
        files_batch_size = 100
        data_path = './disonances/'
        files_disonances = [os.path.join(data_path, file) for file in os.listdir(data_path) if file.endswith(".disonances.npz")]
        files_disonances_dur = [os.path.join(data_path, file) for file in os.listdir(data_path) if file.endswith(".disonances_durations.npz")]
        
        files_not_disonances = [os.path.join(data_path, file) for file in os.listdir(data_path) if file.endswith(".not_disonances.npz")]
        files_not_disonances_dur = [os.path.join(data_path, file) for file in os.listdir(data_path) if file.endswith(".not_disonances_durations.npz")]
        
        #shuffle(files)
        total_files_batches = len(files_disonances) // files_batch_size + 1

        #define model hyperparameters
        config = wandb.config
        config.model = 'bigru'
        config.learning_rate = 0.0001
        config.sequence_length = 20
        config.hidden_values = 200
        config.epochs = 1
        config.batch_size = 32
        
        unique_notes = np.load('./notes_durations/unique_notes.npy', allow_pickle = True)
        unique_dur = np.load('./notes_durations/unique_dur.npy', allow_pickle = True)

        #integer encoding
        note_to_int = dict((note, i) for i, note in enumerate(unique_notes))
        dur_to_int = dict((dur, i) for i, dur in enumerate(unique_dur))
        
        encoded_notes = [note_to_int[note] for note in unique_notes]    #126 (18 game * 7 note)
        encoded_dur = [dur_to_int[dur] for dur in unique_dur]           #5159
        model, _, _ = bilstm.near_disonances_notes_model(config, len(to_categorical(encoded_notes)), len(to_categorical(encoded_dur)))
        model.summary()
        
        #save model to wandb
        # model.save("neural_model.keras")
        # model_artifact = wandb.Artifact("bilstm", type="model", description="Bilstm that will learn the notes near disonances", metadata=dict(config))
        # model_artifact.add_file("neural_model.keras")
        # wandb.save("neural_model.keras")
        
        for file in files_disonances:
            print(i)
            if i == files_batch_size:
                files_batch += 1
                i = 0
                print(len(notes))
                print("Files batch: " + str(files_batch) + " / " + str(total_files_batches))
                # train NN on disonances
                self.prepare_data_and_fit(notes, durations, True, note_to_int, dur_to_int, config, model)
                # train NN on not disonances
                self.prepare_data_and_fit(notes_not_disonances[:len(notes)], durations_not_disonances[:len(durations)], False, note_to_int, dur_to_int, config, model)
                notes = []
                durations = []
                notes_not_disonances = []
                durotions_not_disonances = []
            i += 1

            #load notes for disonances sequences
            disonances_info = np.load(file, allow_pickle = True)
            for data in disonances_info:
                notes.append(disonances_info[data])

            #load durations for disonances sequences
            file2 = file.replace('.npz', '_durations.npz')
            disonances_info = np.load(file2, allow_pickle = True)
            for data in disonances_info:
                durations.append(disonances_info[data])
                
            #load notes for not disonances sequences
            file2 = file.replace('disonances_durations.npz', 'not_disonances.npz')
            disonances_info = np.load(file2, allow_pickle = True)
            for data in disonances_info:
                notes_not_disonances.append(disonances_info[data])

            #load durations for not disonances sequences
            file2 = file.replace('.npz', '_durations.npz')
            disonances_info = np.load(file2, allow_pickle = True)
            for data in disonances_info:
                durations_not_disonances.append(disonances_info[data])

        print("Files batch: " + str(files_batch + 1) + " / " + str(total_files_batches))
        self.prepare_data_and_fit(notes, durations, True, note_to_int, dur_to_int, config, model)
        self.prepare_data_and_fit(notes_not_disonances[:len(notes)], durations_not_disonances[:len(durations)], False, note_to_int, dur_to_int, config, model)
        # wandb.finish()

    def prepare_data_and_fit(self, notes, durations, is_disonance, note_to_int, dur_to_int, config, model):    
        notes_data = []
        durations_data = []
        for i in range(len(notes)):
            notes_data.append([note_to_int.get(note, 0) for note in notes[i]])
            durations_data.append([dur_to_int.get(dur, 0) for dur in durations[i]])
        notes = []
        durations = []
        
        notes_data = to_categorical(notes_data, len(note_to_int))   #notes_data 50000(nr secvente din toate fisierele din file_batch)*20(lungimea secventa)
        durations_data = to_categorical(durations_data, len(dur_to_int))
        
        self.fit(config, notes_data, durations_data, is_disonance, model)

        notes_data = []
        durations_data = []

    #start training
    def fit(self, config, notes_data, durations_data, is_disonance, model):
                
        batch_size = config.batch_size
        for epoch in range(config.epochs):
            print ('epoca: ' + str(epoch + 1))
            test_accuracy = 0
            train_accuracy = 0

            nr_of_batches = len(notes_data) // batch_size 
            for batch_nr in tqdm(range(nr_of_batches)):
                enc_in_data = []
                enc_in_data = durations_data[batch_nr * batch_size : (batch_nr + 1) * batch_size]
                enc_in_data = np.asarray(enc_in_data)

                dec_in_data = []
                dec_in_data = notes_data[batch_nr * batch_size : (batch_nr + 1) * batch_size]
                dec_in_data = np.asarray(dec_in_data)

                if(is_disonance): 
                    dec_out_data = tf.repeat([[1]], repeats = batch_size, axis = 0)
                else: 
                    dec_out_data = tf.repeat([[0]], repeats = batch_size, axis = 0)
                
                loss = model.train_on_batch([enc_in_data, dec_in_data], dec_out_data)
                gc.collect()
                tf.keras.backend.clear_session()
                train_accuracy += loss[1]
#                print ('batch: ' + str(batch_nr + 1) + '/' + str(nr_of_batches) + ' train_loss: ' + str(loss[0]) + ' train_accuracy: ' + str(loss[1]))

            train_accuracy /= nr_of_batches

            # wandb.log({"epochs" : epoch,
            #             "train_accuracy" : train_accuracy
            #             })
            #print ('test accuracy: ' + str(test_accuracy))

        model.save_weights('./' + config.model + '/my_checkpoint')

if __name__ == "__main__":
    obj = DurationsToPitchs()
    obj.start()
