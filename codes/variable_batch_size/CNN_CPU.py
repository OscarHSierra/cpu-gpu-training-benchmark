#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # CPU ONLY

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

BATCH_LIST = [64, 128, 256, 512, 1024, 2048, 4096, 8192,16384]
RUNS = 3
EPOCHS = 4

ALL_FILE = "cnn_cpu_all_runs.csv"
SUMMARY_FILE = "cnn_cpu_summary.csv"

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
# LOAD COMPLETED RUNS
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

print("🖥️ CNN CPU ROBUST BENCHMARK (FINAL)")

# =========================
# DATASET
# =========================
(x_train, y_train), _ = tf.keras.datasets.cifar10.load_data()

x_train = x_train[:50000]
y_train = y_train[:50000]

x_train = x_train.astype("float32") / 255.0

# =========================
# MODEL
# =========================
def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='swish', input_shape=(32,32,3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='swish'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(128, (3,3), padding='same', activation='swish'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dense(128, activation='swish'),
        tf.keras.layers.Dense(10)
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

        for r in range(RUNS):

            if (batch, r+1) in completed:
                print(f"Skipping Batch {batch} Run {r+1}")
                continue

            print(f"Run {r+1}/{RUNS}")

            try:
                tf.keras.backend.clear_session()
                model = build_model()

                # WARM-UP
                model.fit(x_train, y_train, epochs=1, batch_size=batch, verbose=0)

                # MEDICIÓN
                start = time.perf_counter()
                model.fit(x_train, y_train, epochs=EPOCHS, batch_size=batch, verbose=0)
                total_time = time.perf_counter() - start

                time_per_epoch = total_time / EPOCHS
                throughput = len(x_train) / time_per_epoch

                run_times.append(time_per_epoch)
                run_throughputs.append(throughput)

                # GUARDAR RUN
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

print("\n✅ Benchmark finalizado (robusto + reproducible + paper-ready)")