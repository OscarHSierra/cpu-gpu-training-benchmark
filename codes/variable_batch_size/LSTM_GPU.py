#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 00:21:11 2026

@author: dspsogamoso
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
# ❌ NO forzar CPU

import tensorflow as tf
import numpy as np
import time
import random
import csv

# =========================
# CONFIG
# =========================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

BATCH_LIST = [16, 32, 64, 128,256,512]   # ⚠️ más conservador en GPU
RUNS = 3
EPOCHS = 40

ALL_FILE = "lstm_gpu_all_runs.csv"
SUMMARY_FILE = "lstm_gpu_summary.csv"

# =========================
# CONFIG GPU
# =========================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("✅ GPU detectada:", gpus[0])
    except RuntimeError as e:
        print(e)
else:
    print("❌ No GPU detectada")

# =========================
# INIT FILES
# =========================
if not os.path.exists(ALL_FILE):
    with open(ALL_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Batch", "Run",
            "Total Time (s)",
            "Time/Epoch (s)",
            "Throughput (samples/s)"
        ])

if not os.path.exists(SUMMARY_FILE):
    with open(SUMMARY_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Batch",
            "Mean Time/Epoch (s)", "Std Time (s)",
            "Mean Throughput", "Std Throughput"
        ])

# =========================
# LOAD COMPLETED
# =========================
def load_completed():
    done = set()
    if os.path.exists(ALL_FILE):
        with open(ALL_FILE, "r") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                done.add((int(row[0]), int(row[1])))
    return done

completed = load_completed()

print("🚀 LSTM GPU ROBUST BENCHMARK")

# =========================
# DATASET (MISMO QUE CPU)
# =========================
path = tf.keras.utils.get_file(
    'shakespeare.txt',
    'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt'
)

text = open(path, 'rb').read().decode('utf-8')
text = text[:200000]

vocab = sorted(set(text))
char2idx = {u: i for i, u in enumerate(vocab)}
text_as_int = np.array([char2idx[c] for c in text])

# =========================
# DATA PIPELINE
# =========================
SEQ_LENGTH = 80
BUFFER_SIZE = 5000

dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
sequences = dataset.batch(SEQ_LENGTH + 1, drop_remainder=True)

def split_input_target(chunk):
    return chunk[:-1], chunk[1:]

dataset = sequences.map(split_input_target)
dataset = dataset.shuffle(BUFFER_SIZE)

# =========================
# MODEL (MISMO QUE CPU)
# =========================
def build_model(vocab_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, 128),
        tf.keras.layers.LSTM(256, return_sequences=True),
        tf.keras.layers.LSTM(256, return_sequences=True),
        tf.keras.layers.Dense(vocab_size)
    ])

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    )
    return model

# =========================
# EXPERIMENT
# =========================
def run_experiment():

    for batch in BATCH_LIST:

        print(f"\n===== Batch {batch} =====")
        run_times = []
        run_throughputs = []

        dataset_batched = dataset.batch(batch, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

        for r in range(RUNS):

            if (batch, r+1) in completed:
                print(f"Skipping Batch {batch} Run {r+1}")
                continue

            print(f"Run {r+1}/{RUNS}")

            try:
                tf.keras.backend.clear_session()
                model = build_model(len(vocab))

                # WARM-UP
                model.fit(dataset_batched, epochs=1, verbose=0)

                # MEDICIÓN
                start = time.perf_counter()
                model.fit(dataset_batched, epochs=EPOCHS, verbose=0)
                total_time = time.perf_counter() - start

                time_per_epoch = total_time / EPOCHS

                samples = len(text_as_int) // SEQ_LENGTH
                throughput = samples / time_per_epoch

                run_times.append(time_per_epoch)
                run_throughputs.append(throughput)

                # GUARDAR
                with open(ALL_FILE, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        batch,
                        r+1,
                        round(total_time, 4),
                        round(time_per_epoch, 4),
                        round(throughput, 2)
                    ])

                print(f"Total: {total_time:.2f}s | Epoch: {time_per_epoch:.4f}s | Throughput: {throughput:.2f}")

            except Exception as e:
                print(f"❌ Batch {batch} failed: {e}")
                break

        # =========================
        # SUMMARY
        # =========================
        if run_times:
            mean_time = np.mean(run_times)
            std_time = np.std(run_times)

            mean_tp = np.mean(run_throughputs)
            std_tp = np.std(run_throughputs)

            print(f">>> Batch {batch}:")
            print(f"    Time: {mean_time:.4f} ± {std_time:.4f}")
            print(f"    Throughput: {mean_tp:.2f} ± {std_tp:.2f}")

            with open(SUMMARY_FILE, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    batch,
                    round(mean_time, 4),
                    round(std_time, 4),
                    round(mean_tp, 2),
                    round(std_tp, 2)
                ])

# =========================
# RUN
# =========================
run_experiment()

print("\n✅ LSTM GPU Benchmark finalizado")


