import os, sys
from music21 import converter, instrument, note, chord
import numpy as np

import chromosome_transformer

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Model
from tqdm import tqdm
from random import shuffle
import wandb
import gc

# wandb.login()
# wandb.init(project='neuralEsthetics', entity='razvan')

os.environ["CUDA_VISIBLE_DEVICES"] = ""
# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#     try:
#         tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2000)])
#     except RuntimeError as e:
#         print(e)
    
class NotesGenerator:

    #load data from preprocessed files
    def start(self):        
        files_batch_size = 100
        data_path = './notes_durations/'
        files = [os.path.join(data_path, file) for file in os.listdir(data_path) if file.endswith("notes.npy")]
        shuffle(files)
        files.remove('./notes_durations/unique_notes.npy')

        #files = files[:2]
        
        total_files_batches = len(files) // files_batch_size + 1

        #define model hyperparameters
        config = wandb.config
        config.model = 'chromosome_transformer'
        config.learning_rate = 0.0001
        config.sequence_length = 50
        #config.hidden_values = 500
        config.epochs = 10
        config.batch_size = 64
        config.embed_dim = 256
        config.num_layers = 1
        
        unique_notes = np.load('./notes_durations/unique_notes.npy', allow_pickle = True)
        unique_dur = np.load('./notes_durations/unique_dur.npy', allow_pickle = True)

        #integer encoding
        note_to_int = dict((note, i) for i, note in enumerate(unique_notes))
        dur_to_int = dict((dur, i) for i, dur in enumerate(unique_dur))
        
        encoded_notes = [note_to_int[note] for note in unique_notes]    #126 (18 game * 7 note)
        encoded_dur = [dur_to_int[dur] for dur in unique_dur]           #5159
        model = chromosome_transformer.transformer_model(config, len(note_to_int), len(dur_to_int))
        #model.summary()

        #save model to wandb
        # model.save("neural_model.keras", include_optimizer = False)
        # model_artifact = wandb.Artifact("transformer", type="model", 
        #                                 description="Transformer model used to generate next notes based on current ones", 
        #                                 metadata=dict(config))
        # model_artifact.add_file("neural_model.keras")
        # wandb.save("neural_model.keras")
        
        for epoch in range(config.epochs):
            notes = []
            durations = []
            i = 0
            files_batch = 0
            
            for file in files:
                if i == files_batch_size:
                    files_batch += 1
                    i = 0
                    print ('Epoch: ' + str(epoch + 1))
                    print("Files batch: " + str(files_batch) + " / " + str(total_files_batches))
                    print("Notes to learn: " + str(len(notes)))
                    self.prepare_data_and_fit(notes, durations, note_to_int, dur_to_int, config, model, epoch)
                    notes = []
                    durations = []
                i += 1
    
                #load durations
                bach_info = np.load(file, allow_pickle = True)
                for data in bach_info:
                    notes.append(data)
    
                #load notes
                bach_info = np.load(file.replace('.notes.npy', '.durations.npy'), allow_pickle = True)
                for data in bach_info:
                    durations.append(data)
    
            print ('Epoch: ' + str(epoch + 1))
            print("Files batch: " + str(files_batch + 1) + " / " + str(total_files_batches))
            print("Notes to learn: " + str(len(notes)))
            self.prepare_data_and_fit(notes, durations, note_to_int, dur_to_int, config, model, epoch)
        # wandb.finish()

    def prepare_data_and_fit(self, notes, durations, note_to_int, dur_to_int, config, model, epoch):        
        notes_data = []
        durations_data = []
        for i in range(0, len(notes) - config.sequence_length):
            notes_data.append([note_to_int.get(note, 0) for note in notes[i : i + config.sequence_length]])
            durations_data.append([dur_to_int.get(dur, 0) for dur in durations[i : i + config.sequence_length]])
        notes = []
        durations = []
        
        #notes_data = to_categorical(notes_data, len(note_to_int))   #notes_data 50000(nr secvente din toate fisierele din file_batch)*20(lungimea secventa)
        #durations_data = to_categorical(durations_data, len(dur_to_int))

        notes_test_data = notes_data[len(notes_data) * 6 // 10 : len(notes_data)]
        notes_data = notes_data[:len(notes_data) * 6 // 10]
        durations_test_data = durations_data[len(durations_data) * 6 // 10 : len(durations_data)]
        durations_data = durations_data[:len(durations_data) * 6 // 10]

        self.fit(config, notes_data, durations_data, notes_test_data, durations_test_data, model, epoch)

        notes_test_data = []
        notes_data = []
        # durations_test_data = []
        # durations_data = []

    #start training
    def fit(self, config, notes_data, durations_data, notes_test_data, durations_test_data, model, epoch):
                
        batch_size = config.batch_size
        test_accuracy = 0
        train_accuracy = 0

        nr_of_batches = len(notes_data) // batch_size 
        for batch_nr in tqdm(range(nr_of_batches)):
            notes_batch_data = []
            notes_batch_data = notes_data[batch_nr * batch_size : (batch_nr + 1) * batch_size]
            
            dur_batch_data = []
            dur_batch_data = durations_data[batch_nr * batch_size : (batch_nr + 1) * batch_size]

            enc1_in_data = notes_batch_data
            enc1_in_data = np.delete(enc1_in_data, -1, 1) # erase the first column of data
            enc1_in_data = np.hstack((np.zeros((batch_size, 1)), enc1_in_data))
            enc1_in_data = np.asarray(enc1_in_data)

            enc2_in_data = dur_batch_data
            enc2_in_data = np.delete(enc2_in_data, -1, 1) # erase the first column of data
            enc2_in_data = np.hstack((np.zeros((batch_size, 1)), enc2_in_data))
            enc2_in_data = np.asarray(enc2_in_data)
            
            dec1_out_data = np.asarray(notes_batch_data)
            dec2_out_data = np.asarray(dur_batch_data)
            
            loss = model.train_on_batch([dec1_out_data, dec2_out_data], [dec1_out_data, dec2_out_data])
            gc.collect()
            tf.keras.backend.clear_session()
            train_accuracy += (loss[3] + loss[4]) / 2
#                print ('batch: ' + str(batch_nr + 1) + '/' + str(nr_of_batches) + ' train_loss: ' + str(loss[0]) + ' train_accuracy: ' + str(loss[1]))

        train_accuracy /= nr_of_batches
        nr_of_batches = len(notes_test_data) // batch_size
        for batch_nr in tqdm(range(nr_of_batches)):
            notes_batch_data = []
            notes_batch_data = notes_test_data[batch_nr * batch_size : (batch_nr + 1) * batch_size]
            
            dur_batch_data = []
            dur_batch_data = durations_test_data[batch_nr * batch_size : (batch_nr + 1) * batch_size]

            enc1_in_data = notes_batch_data
            enc1_in_data = np.delete(enc1_in_data, -1, 1) # erase the first column of data
            enc1_in_data = np.hstack((np.zeros((batch_size, 1)), enc1_in_data))
            enc1_in_data = np.asarray(enc1_in_data)

            enc2_in_data = dur_batch_data
            enc2_in_data = np.delete(enc2_in_data, -1, 1) # erase the first column of data
            enc2_in_data = np.hstack((np.zeros((batch_size, 1)), enc2_in_data))
            enc2_in_data = np.asarray(enc2_in_data)
            
            dec1_out_data = np.asarray(notes_batch_data)
            dec2_out_data = np.asarray(dur_batch_data)


            loss = model.test_on_batch([dec1_out_data, dec2_out_data], [dec1_out_data, dec2_out_data])
            print('\n' + str(loss))
            gc.collect()
            tf.keras.backend.clear_session()
            test_accuracy += (loss[3] + loss[4]) / 2

        test_accuracy /= nr_of_batches
        # wandb.log({"epochs" : epoch,
        #            "train_accuracy" : train_accuracy,
        #            "test_accuracy" : test_accuracy
        #            })
        print ('test accuracy: ' + str(test_accuracy))

        model.save_weights('./' + config.model + '/my_checkpoint')

if __name__ == "__main__":
    obj = NotesGenerator()
    obj.start()
