import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
from conformer_block import ConformerBlock

# --- CONFIGURATION ---
TFRECORD_DIR = "./tfrecords"
MODEL_PATH = "./models/piano_transcription_model_conformer (6).h5"
N_BINS = 288
NUM_PIANO_KEYS = 88
BATCH_SIZE = 16
SEGMENT_SECONDS = 15.0
SR = 16000
HOP_SIZE = 512
SEGMENT_FRAMES = int(np.ceil(SEGMENT_SECONDS * SR / HOP_SIZE))

# --- TFRecord parsing ---
def _parse_example(example_proto):
    feature_description = {
        'cqt': tf.io.VarLenFeature(tf.float32),
        'active_notes_labels': tf.io.VarLenFeature(tf.int64),
        'attack_labels': tf.io.VarLenFeature(tf.int64),
        'release_labels': tf.io.VarLenFeature(tf.int64),
        'spec_shape': tf.io.FixedLenFeature([3], tf.int64),
        'labels_shape': tf.io.FixedLenFeature([2], tf.int64),
        'length': tf.io.FixedLenFeature([1], tf.int64)
    }
    features = tf.io.parse_single_example(example_proto, feature_description)
    spec = tf.reshape(tf.sparse.to_dense(features['cqt']), features['spec_shape'])
    labels_shape = features['labels_shape']
    active_notes = tf.cast(tf.reshape(tf.sparse.to_dense(features['active_notes_labels']), labels_shape), tf.float32)
    attacks = tf.cast(tf.reshape(tf.sparse.to_dense(features['attack_labels']), labels_shape), tf.float32)
    releases = tf.cast(tf.reshape(tf.sparse.to_dense(features['release_labels']), labels_shape), tf.float32)
    length = tf.cast(features['length'][0], tf.int32)
    return spec, active_notes, attacks, releases, length

def slice_and_format_for_test(spec, active_notes, attacks, releases, length):
    spec_slice = spec[:SEGMENT_FRAMES, :, :]
    active_notes_slice = active_notes[:SEGMENT_FRAMES, :]
    attacks_slice = attacks[:SEGMENT_FRAMES, :]
    releases_slice = releases[:SEGMENT_FRAMES, :]
    pad_amt = SEGMENT_FRAMES - tf.shape(spec_slice)[0]
    spec_slice = tf.pad(spec_slice, [[0, pad_amt], [0, 0], [0, 0]], constant_values=-80)
    active_notes_slice = tf.pad(active_notes_slice, [[0, pad_amt], [0, 0]])
    attacks_slice = tf.pad(attacks_slice, [[0, pad_amt], [0, 0]])
    releases_slice = tf.pad(releases_slice, [[0, pad_amt], [0, 0]])
    final_spec = tf.ensure_shape(spec_slice, [SEGMENT_FRAMES, N_BINS, 1])
    return final_spec, {
        'attack_output': attacks_slice,
        'active_notes_output': active_notes_slice,
        'release_output': releases_slice
    }

def create_test_dataset(tfrecord_path):
    dataset = tf.data.TFRecordDataset(tfrecord_path, num_parallel_reads=tf.data.AUTOTUNE)
    dataset = dataset.map(_parse_example, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(slice_and_format_for_test, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

if __name__ == '__main__':
    test_dataset = create_test_dataset(os.path.join(TFRECORD_DIR, 'test.tfrecord'))
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects={'ConformerBlock': ConformerBlock})

    print("Evaluating on the test set...")
    results = model.evaluate(test_dataset, return_dict=True)
    print("\n--- Test Evaluation Results ---")
    for key, value in results.items():
        print(f"{key}: {value:.4f}")
