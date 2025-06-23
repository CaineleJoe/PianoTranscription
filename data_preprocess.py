import pandas as pd
import numpy as np
import librosa
import pretty_midi
import os
import tensorflow as tf
from tqdm import tqdm
from note_seq import sequences_lib, midi_io
import multiprocessing

# --- CONFIG ---
DATASET_PATH = "D:/music/maestro-v3.0.0"
CSV_PATH = os.path.join(DATASET_PATH, "maestro-v3.0.0.csv")
OUTPUT_DIR = "./tfrecords"

SR = 16000
HOP_SIZE = 512
N_BINS = 288
BINS_PER_OCTAVE = 36
FMIN = 27.5
PIANO_MIN_MIDI, PIANO_MAX_MIDI = 21, 108
NUM_PIANO_KEYS = (PIANO_MAX_MIDI - PIANO_MIN_MIDI) + 1

WINDOW_SEC = 15.0
OVERLAP = 0.5
STRIDE_SEC = WINDOW_SEC * (1 - OVERLAP)

def _float_feature(value): return tf.train.Feature(float_list=tf.train.FloatList(value=value))
def _int64_feature(value): return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
def _bytes_feature(value): return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def create_tf_example(cqt, active_notes_labels, attack_labels, release_labels, length):
    feature = {
        'cqt': _float_feature(cqt.flatten().tolist()),
        'active_notes_labels': _int64_feature(active_notes_labels.flatten().tolist()),
        'attack_labels': _int64_feature(attack_labels.flatten().tolist()),
        'release_labels': _int64_feature(release_labels.flatten().tolist()),
        'length': _int64_feature([length]),
        'spec_shape': _int64_feature(list(cqt.shape)),
        'labels_shape': _int64_feature(list(active_notes_labels.shape)),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

def process_file(args):
    audio_path, midi_path, split, is_fragmented = args
    results = []
    try:
        y, _ = librosa.load(audio_path, sr=SR)
        cqt = librosa.cqt(y, sr=SR, hop_length=HOP_SIZE, fmin=FMIN, n_bins=N_BINS, bins_per_octave=BINS_PER_OCTAVE)
        cqt = np.abs(cqt).astype(np.float32)
        cqt = librosa.amplitude_to_db(cqt, ref=np.max)
        cqt = np.nan_to_num(cqt, nan=-80, neginf=-80)
        cqt = np.clip(cqt, a_min=-80, a_max=None)
        cqt = cqt.T
        cqt = cqt[..., np.newaxis]

        midi_data = pretty_midi.PrettyMIDI(midi_path)
        sequence = midi_io.midi_to_note_sequence(midi_data)
        sustained = sequences_lib.apply_sustain_control_changes(sequence)

        num_frames = cqt.shape[0]
        window_frames = int(round(WINDOW_SEC * SR / HOP_SIZE))
        stride_frames = int(round(STRIDE_SEC * SR / HOP_SIZE))

        active_roll = np.zeros((NUM_PIANO_KEYS, num_frames), dtype=np.uint8)
        attack_roll = np.zeros((NUM_PIANO_KEYS, num_frames), dtype=np.uint8)
        release_roll = np.zeros((NUM_PIANO_KEYS, num_frames), dtype=np.uint8)
        for note in sustained.notes:
            if PIANO_MIN_MIDI <= note.pitch <= PIANO_MAX_MIDI:
                pitch_idx = note.pitch - PIANO_MIN_MIDI
                start_frame = librosa.time_to_frames(note.start_time, sr=SR, hop_length=HOP_SIZE)
                end_frame = librosa.time_to_frames(note.end_time, sr=SR, hop_length=HOP_SIZE)
                if start_frame < num_frames:
                    attack_roll[pitch_idx, start_frame] = 1
                if end_frame < num_frames:
                    release_roll[pitch_idx, end_frame] = 1
                for f in range(start_frame, min(end_frame, num_frames)):
                    active_roll[pitch_idx, f] = 1
        active_notes_labels_all = active_roll.T
        attack_labels_all = attack_roll.T
        release_labels_all = release_roll.T

        if is_fragmented:
            for start in range(0, num_frames, stride_frames):
                end = start + window_frames
                frag_cqt = cqt[start:end]
                frag_active_notes_labels = active_notes_labels_all[start:end]
                frag_attack_labels = attack_labels_all[start:end]
                frag_release_labels = release_labels_all[start:end]
                length = frag_cqt.shape[0]
                if length < window_frames:
                    pad_width = window_frames - length
                    frag_cqt = np.pad(frag_cqt, ((0, pad_width), (0,0), (0,0)), constant_values=-80)
                    frag_active_notes_labels = np.pad(frag_active_notes_labels, ((0, pad_width), (0,0)), constant_values=0)
                    frag_attack_labels = np.pad(frag_attack_labels, ((0, pad_width), (0,0)), constant_values=0)
                    frag_release_labels = np.pad(frag_release_labels, ((0, pad_width), (0,0)), constant_values=0)
                example = create_tf_example(frag_cqt, frag_active_notes_labels, frag_attack_labels, frag_release_labels, length)
                results.append(example.SerializeToString())
        else:
            frag_cqt = cqt
            frag_active_notes_labels = active_notes_labels_all
            frag_attack_labels = attack_labels_all
            frag_release_labels = release_labels_all
            length = frag_cqt.shape[0]
            example = create_tf_example(frag_cqt, frag_active_notes_labels, frag_attack_labels, frag_release_labels, length)
            results.append(example.SerializeToString())
    except Exception as e:
        print(f"EROARE la {audio_path}: {e}")
    return results

def main():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    df = pd.read_csv(CSV_PATH)
    splits = ['train', 'validation', 'test']

    num_processes = max(1, os.cpu_count() - 1)
    print(f"Se folosesc {num_processes} procese paralele.")

    for split in splits:
        print(f"\n=== PRELUCRARE SPLIT: {split.upper()} ===")
        split_df = df[df['split'] == split]
        tasks = []
        for idx, row in split_df.iterrows():
            audio_path = os.path.join(DATASET_PATH, row['audio_filename'])
            midi_path = os.path.join(DATASET_PATH, row['midi_filename'])
            if os.path.exists(audio_path) and os.path.exists(midi_path):
                is_fragmented = (split != 'test')
                tasks.append((audio_path, midi_path, split, is_fragmented))

        output_tfrecord_path = os.path.join(OUTPUT_DIR, f"{split}.tfrecord")
        with tf.io.TFRecordWriter(output_tfrecord_path) as writer:
            with multiprocessing.Pool(processes=num_processes) as pool:
                progress_bar = tqdm(pool.imap_unordered(process_file, tasks), total=len(tasks), desc=f"Procesare {split}")
                for result_list in progress_bar:
                    for serialized_example in result_list:
                        writer.write(serialized_example)
        print(f"Saved: {output_tfrecord_path}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
