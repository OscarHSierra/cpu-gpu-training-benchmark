import tensorflow as tf
import numpy as np
import random
import os
import time

# Fijar semilla global
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

def run_training(device_name):
    print(f"\nEntrenando en: {device_name}")
    tf.keras.backend.clear_session()
    with tf.device(device_name):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        x_train = x_train.astype("float32") / 255.0
        x_test = x_test.astype("float32") / 255.0

        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='swish', input_shape=(32, 32, 3)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='swish'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.3),

            tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='swish'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='swish'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.4),

            tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='swish'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='swish'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.GlobalAveragePooling2D(),

            tf.keras.layers.Dense(1024, activation='swish'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(512, activation='swish'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(10)
        ])

        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        start_time = time.time()
        model.fit(x_train, y_train, epochs=1, batch_size=512, verbose=2)
        elapsed_time = time.time() - start_time
        print(f" Tiempo de entrenamiento en {device_name}: {elapsed_time:.2f} segundos")
        return elapsed_time
cpu_time = run_training("/CPU:0")
cpu_time = run_training("/CPU:0")
cpu_time = run_training("/CPU:0")
cpu_time = run_training("/CPU:0")
if tf.config.list_physical_devices('GPU'):
    gpu_time = run_training("/GPU:0")
    gpu_time = run_training("/GPU:0")
    gpu_time = run_training("/GPU:0")
    gpu_time = run_training("/GPU:0")
else:
    print("\n️ GPU no disponible.")


