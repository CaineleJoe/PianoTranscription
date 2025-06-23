import numpy as np
import tensorflow as tf
import librosa
from note_seq import midi_io, sequences_lib
from conformer_block import ConformerBlock

# --------- CONFIG ---------
MODEL_PATH = "models/piano_transcription_model_conformer (6).h5"
WAV_PATH = "C:/Users/Luca/Downloads/Beethoven - Moonlight Sonata (FULL).mp3"
OUTPUT_MIDI = "predict_output.mid"

SR = 16000
HOP_SIZE = 512
N_BINS = 288
BINS_PER_OCTAVE = 36
FMIN = 27.5
SEGMENT_SECONDS = 15.0
PIANO_MIN_MIDI, PIANO_MAX_MIDI = 21, 108
SEGMENT_FRAMES = int(np.ceil(SEGMENT_SECONDS * SR / HOP_SIZE))

def wav_to_cqt(filepath):
    y, sr = librosa.load(filepath, sr=SR, mono=True)
    cqt = librosa.cqt(
        y,
        sr=SR,
        hop_length=HOP_SIZE,
        fmin=FMIN,
        n_bins=N_BINS,
        bins_per_octave=BINS_PER_OCTAVE
    )
    cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
    cqt_db = np.clip(cqt_db, -80, 0)
    cqt_db = cqt_db.T
    cqt_db = cqt_db[..., np.newaxis]
    print("CQT shape:", cqt_db.shape)
    return cqt_db.astype(np.float32)

def run_model(model, cqt_db, segment_frames=469, batch_size=1):
    num_frames = cqt_db.shape[0]
    segments = []
    for start in range(0, num_frames, segment_frames):
        end = min(start + segment_frames, num_frames)
        seg = cqt_db[start:end]
        if seg.shape[0] < segment_frames:
            pad = segment_frames - seg.shape[0]
            seg = np.pad(seg, ((0, pad), (0, 0), (0, 0)), constant_values=-80)
        segments.append(seg)
    segments = np.stack(segments)

    preds = model.predict(segments, batch_size=batch_size)
    attack_probs = np.concatenate([p for p in preds['attack_output']], axis=0)[:num_frames, :]
    active_notes_probs = np.concatenate([p for p in preds['active_notes_output']], axis=0)[:num_frames, :]
    release_probs = np.concatenate([p for p in preds['release_output']], axis=0)[:num_frames, :]
    return attack_probs, active_notes_probs, release_probs

def probs_to_notes(active_notes_probs, attack_probs, threshold=0.5):
    active_notes_pred = (active_notes_probs > threshold).astype(np.int32)
    attack_pred = (attack_probs > threshold).astype(np.int32)
    frames_per_second = SR / HOP_SIZE
    seq = sequences_lib.pianoroll_to_note_sequence(
        frames=active_notes_pred,
        frames_per_second=frames_per_second,
        min_duration_ms=0,
        min_midi_pitch=PIANO_MIN_MIDI,
        onset_predictions=attack_pred
    )
    return seq

if __name__ == "__main__":
    print("Loading model...")
    model = tf.keras.models.load_model(
        MODEL_PATH, compile=False, custom_objects={"ConformerBlock": ConformerBlock}
    )
    print("Processing audio...")
    cqt_db = wav_to_cqt(WAV_PATH)
    print("Running model prediction...")
    attack_probs, active_notes_probs, release_probs = run_model(
        model, cqt_db, segment_frames=SEGMENT_FRAMES, batch_size=1)
    print("Generating MIDI sequence...")
    pred_seq = probs_to_notes(active_notes_probs, attack_probs, threshold=0.45)
    print(f"Saving prediction as MIDI: {OUTPUT_MIDI}")
    midi_io.sequence_proto_to_midi_file(pred_seq, OUTPUT_MIDI)
    print("Done.")
