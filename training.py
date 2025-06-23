import tensorflow as tf
from tensorflow.keras import layers, models
import os
import numpy as np
from conformer_block import ConformerBlock

# --- CONFIGURATION ---
TFRECORD_DIR = "/kaggle/input/maestro-dataset-preprocessed/"
MODEL_SAVE_PATH = "/kaggle/working/piano_transcription_model_conformer.h5"

N_BINS = 288
SR = 16000
HOP_SIZE = 512
NUM_PIANO_KEYS = 88
BATCH_SIZE = 16
SEGMENT_SECONDS = 15.0
SEGMENT_FRAMES = int(np.ceil(SEGMENT_SECONDS * SR / HOP_SIZE))

# --- TFRecord data ---
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

def create_dataset(tfrecord_path, is_training=True):
    dataset = tf.data.TFRecordDataset(tfrecord_path, num_parallel_reads=tf.data.AUTOTUNE)
    dataset = dataset.map(_parse_example, num_parallel_calls=tf.data.AUTOTUNE)
    if is_training:
        dataset = dataset.repeat()
        dataset = dataset.shuffle(buffer_size=256)
        dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    else:
        dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

def build_piano_transcription_conformer(input_shape, num_pitches, hparams):
    model_input = layers.Input(shape=input_shape)  # (frames, bins, 1)
    # CNN Feature extractor
    x = layers.Conv2D(48, (3, 3), padding='same', activation='relu')(model_input)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((1, 2), padding='same')(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Conv2D(48, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((1, 2), padding='same')(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Conv2D(96, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((1, 2), padding='same')(x)
    x = layers.Dropout(0.25)(x)

    x_flat = layers.Reshape((SEGMENT_FRAMES, -1))(x)
    x_proj = layers.Dense(hparams['conformer_dim'])(x_flat)

    # --- Conformer blocks stack ---
    x_conf = x_proj
    for _ in range(hparams['num_conformer_blocks']):
        x_conf = ConformerBlock(
            d_model=hparams['conformer_dim'],
            num_heads=hparams['conformer_heads'],
            conv_kernel_size=31,
            dropout=hparams['conformer_dropout']
        )(x_conf)

    attack_output = layers.Dense(num_pitches, activation='sigmoid', name="attack_output")(x_conf)
    release_output = layers.Dense(num_pitches, activation='sigmoid', name="release_output")(x_conf)
    active_notes_output = layers.Dense(num_pitches, activation='sigmoid', name="active_notes_output")(x_conf)

    model = models.Model(
        inputs=model_input,
        outputs={
            'attack_output': attack_output,
            'active_notes_output': active_notes_output,
            'release_output': release_output
        },
        name="PianoTranscription_Conformer"
    )
    return model

def train_model():
    input_shape = (SEGMENT_FRAMES, N_BINS, 1)
    hparams = {
        'conformer_dim': 192,
        'conformer_heads': 6,
        'conformer_dropout': 0.3,
        'num_conformer_blocks': 2
    }
    model = build_piano_transcription_conformer(input_shape, NUM_PIANO_KEYS, hparams)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0004, clipnorm=1.0)
    model.compile(
        optimizer=optimizer,
        loss={
            'attack_output': tf.keras.losses.BinaryCrossentropy(from_logits=False),
            'active_notes_output': tf.keras.losses.BinaryCrossentropy(from_logits=False),
            'release_output': tf.keras.losses.BinaryCrossentropy(from_logits=False)
        },
        metrics={
            'attack_output': tf.keras.metrics.BinaryAccuracy(),
            'active_notes_output': tf.keras.metrics.BinaryAccuracy(),
            'release_output': tf.keras.metrics.BinaryAccuracy()
        }
    )
    model.summary()
    print("Creating data pipelines...")
    train_dataset = create_dataset(os.path.join(TFRECORD_DIR, 'train.tfrecord'))
    validation_dataset = create_dataset(os.path.join(TFRECORD_DIR, 'validation.tfrecord'), is_training=False)

    validation_steps = 400
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor='val_loss', verbose=1),
        tf.keras.callbacks.EarlyStopping(patience=4, monitor='val_loss', verbose=1, restore_best_weights=True)
    ]
    print("Starting training...")
    model.fit(
        train_dataset,
        epochs=500,
        validation_data=validation_dataset,
        steps_per_epoch=1000,
        validation_steps=validation_steps,
        callbacks=callbacks
    )

if __name__ == '__main__':
    if not os.path.exists(TFRECORD_DIR):
        print(f"ERROR: TFRecord directory not found: {TFRECORD_DIR}")
    else:
        train_model()
