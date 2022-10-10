import os
from music21 import converter, instrument, note, interval, chord
import numpy as np
#from joblib import Parallel, delayed

def _extract_midi_info(f, d_path, d, i):
    #if not (os.path.isfile(d_path + f.replace('mp3', 'npy'))):
    print(i)
    try:
        midi = converter.parse(f)

        notes_to_parse = midi.flat.notes #.parts.stream()
       
        #eviti erori de genul 
        #'Chord' object has no attribute 'diatonicNoteNum'
        #'Chord' object has no attribute 'pitch'
        #in comparatie cu
        #notes_to_parse = midi.parts.stream().flat.notes
        
        # notes_to_parse = None
        # try: # fisierul midi are mai multe instrumente care canta
        #     s2 = instrument.partitionByInstrument(midi)
        #     notes_to_parse = s2.parts[0].recurse()
        # except: # file has notes in a flat structure
        #     notes_to_parse = midi.flat.notes
        
        notes = []
        durations = []

        nr_notes_near_disonance = 10
        j = 1
        
        disonanceNr = 0
        disonances = {}
        disonances_durations = {}
        
        not_disonanceNr = 0
        not_disonances = {}
        not_disonances_durations = {}
        for element in notes_to_parse[1:]: # pt fiecare nota din melodia curenta
            if isinstance(element, note.Note):
                
                if(j - nr_notes_near_disonance >= 0 and j + nr_notes_near_disonance < len(notes_to_parse)):
                    #colectez toate intervalele disonante si ale 10 note din jurul lor
                    #pe randul urmatore intervalul (dintre nota curenta si cea precedenta) este o disonanta
                    if interval.Interval(noteStart = notes_to_parse[j - 1], noteEnd = element).simpleName in ["m2", "M2", "TT", "m7", "M7"]: 
                        disonances[str(disonanceNr)] = [str(n.pitch) for n in notes_to_parse[j - nr_notes_near_disonance : j + nr_notes_near_disonance]]
                        disonances_durations[str(disonanceNr)] = [str(n.duration.quarterLength) for n in notes_to_parse[j - nr_notes_near_disonance : j + nr_notes_near_disonance]]
                        disonanceNr += 1
                    else:
                        not_disonances[str(not_disonanceNr)] = [str(n.pitch) for n in notes_to_parse[j - nr_notes_near_disonance : j + nr_notes_near_disonance]]
                        not_disonances_durations[str(not_disonanceNr)] = [str(n.duration.quarterLength) for n in notes_to_parse[j - nr_notes_near_disonance : j + nr_notes_near_disonance]]
                        not_disonanceNr += 1
                #colectez toate notele si duratele
                notes.append(str(element.pitch)) # clasa Note a Music21 contine multe proprietati si metode
                durations.append(str(element.duration.quarterLength))
            # elif isinstance(element, chord.Chord): #pt moment evit disonantele dintre acorduri
            #     current_note = element.root() #nota radacina a acordului
                
            j += 1

        #tratez dict disonances ca o matrice (fiecare linie = values pt fiecare key)
        np.savez(d_path + str(i) + '.disonances.npz', **disonances)
        np.savez(d_path + str(i) + '.disonances_durations.npz', **disonances_durations)
        
        np.savez(d_path + str(i) + '.not_disonances.npz', **not_disonances)
        np.savez(d_path + str(i) + '.not_disonances_durations.npz', **not_disonances_durations)
        
        try:
            unique_notes = set(np.load(d + "unique_notes.npy", allow_pickle = True))
            unique_dur = set(np.load(d + "unique_dur.npy", allow_pickle = True))
        except Exception as e:
            print(e)
#        print(unique_notes)
        prev_n = len(unique_notes)
        prev_d = len(unique_dur)
        
        labels = set(notes)
        unique_notes = unique_notes.union(labels)
        
        labels = set(durations)
        unique_dur = unique_dur.union(labels)

        np.save(d + 'unique_notes', list(unique_notes))
        np.save(d + 'unique_dur', list(unique_dur))

        np.save(d + str(i) + '.notes', notes)
        np.save(d + str(i) + '.durations', durations)
    except Exception as e: print (e)
    #else: print ('exist')

def generate_normalized_data(): 
        s_path = './lmd_full/'
        d_path = './disonances/'
        d_path_notes_dur = './notes_durations/'
        #d_path = './feature/cqt/'
        f = []

        for root, dirs, files in os.walk(os.path.expanduser(s_path)):
            for file in files:
                if file.endswith(".mid"):
                    f.append(os.path.join(root, file))
        
        if not os.path.exists(d_path):
                os.makedirs(d_path)
        if not os.path.exists(d_path_notes_dur):
                os.makedirs(d_path_notes_dur)

        np.save(d_path_notes_dur + 'unique_notes', list(set()))
        np.save(d_path_notes_dur + 'unique_dur', list(set()))
        
        for i, mp3_file in enumerate(f[:10]):
              _extract_midi_info(mp3_file, d_path, d_path_notes_dur, i)
            
        #Parallel(n_jobs = 10)(delayed(_extract_midi_info)(mp3_file, d_path, i) for i, mp3_file in enumerate(f))
        
generate_normalized_data()

    


