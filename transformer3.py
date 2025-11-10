import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Para reproducibilidad
np.random.seed(42)
tf.random.set_seed(42)

# Generar datos sintéticos medianos
def generate_synthetic_data(num_samples=20000, seq_len=100, vocab_size=5000, num_classes=5):
    x = np.random.randint(1, vocab_size, size=(num_samples, seq_len))
    y = np.random.randint(0, num_classes, size=(num_samples,))
    return x, y

# Bloque transformer simple
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.1):
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Dense(ff_dim, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(inputs.shape[-1])(x)
    return x + res

# Modelo transformer moderado
def build_model(input_shape, vocab_size, num_classes):
    inputs = keras.Input(shape=input_shape)
    x = layers.Embedding(input_dim=vocab_size, output_dim=64)(inputs)

    # 2 bloques transformer
    for _ in range(2):
        x = transformer_encoder(x, head_size=64, num_heads=4, ff_dim=128, dropout=0.1)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)

# Preparar datos
x_train, y_train = generate_synthetic_data()
x_val, y_val = generate_synthetic_data(num_samples=2000)

#Entrenamiento en CPU
with tf.device("/CPU:0"):
    model_cpu = build_model(input_shape=(100,), vocab_size=5000, num_classes=5)
    model_cpu.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    start_cpu = time.time()
    model_cpu.fit(x_train, y_train, epochs=3, batch_size=1024, validation_data=(x_val, y_val), verbose=1)
    cpu_time = time.time() - start_cpu
    print(f" Tiempo total en CPU: {cpu_time:.2f} segundos")
    #Entrenamiento en CPU
with tf.device("/CPU:0"):
    model_cpu = build_model(input_shape=(100,), vocab_size=5000, num_classes=5)
    model_cpu.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    start_cpu = time.time()
    model_cpu.fit(x_train, y_train, epochs=3, batch_size=1024, validation_data=(x_val, y_val), verbose=1)
    cpu_time = time.time() - start_cpu
    print(f"️ Tiempo total en CPU: {cpu_time:.2f} segundos")
    #Entrenamiento en CPU
with tf.device("/CPU:0"):
    model_cpu = build_model(input_shape=(100,), vocab_size=5000, num_classes=5)
    model_cpu.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    start_cpu = time.time()
    model_cpu.fit(x_train, y_train, epochs=3, batch_size=1024, validation_data=(x_val, y_val), verbose=1)
    cpu_time = time.time() - start_cpu
    print(f"️ Tiempo total en CPU: {cpu_time:.2f} segundos")
    #Entrenamiento en CPU
with tf.device("/CPU:0"):
    model_cpu = build_model(input_shape=(100,), vocab_size=5000, num_classes=5)
    model_cpu.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    start_cpu = time.time()
    model_cpu.fit(x_train, y_train, epochs=3, batch_size=1024, validation_data=(x_val, y_val), verbose=1)
    cpu_time = time.time() - start_cpu
    print(f"️ Tiempo total en CPU: {cpu_time:.2f} segundos")
    #Entrenamiento en CPU
with tf.device("/CPU:0"):
    model_cpu = build_model(input_shape=(100,), vocab_size=5000, num_classes=5)
    model_cpu.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    start_cpu = time.time()
    model_cpu.fit(x_train, y_train, epochs=3, batch_size=1024, validation_data=(x_val, y_val), verbose=1)
    cpu_time = time.time() - start_cpu
    print(f"️ Tiempo total en CPU: {cpu_time:.2f} segundos")

# Entrenamiento en GPU si está disponible
if tf.config.list_physical_devices("GPU"):
    with tf.device("/GPU:0"):
        model_gpu = build_model(input_shape=(100,), vocab_size=5000, num_classes=5)
        model_gpu.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        start_gpu = time.time()
        model_gpu.fit(x_train, y_train, epochs=3, batch_size=1024, validation_data=(x_val, y_val), verbose=1)
        gpu_time = time.time() - start_gpu
        print(f" Tiempo total en GPU: {gpu_time:.2f} segundos")
else:
    print(" GPU no detectada")
# Entrenamiento en GPU si está disponible
if tf.config.list_physical_devices("GPU"):
    with tf.device("/GPU:0"):
        model_gpu = build_model(input_shape=(100,), vocab_size=5000, num_classes=5)
        model_gpu.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        start_gpu = time.time()
        model_gpu.fit(x_train, y_train, epochs=3, batch_size=1024, validation_data=(x_val, y_val), verbose=1)
        gpu_time = time.time() - start_gpu
        print(f" Tiempo total en GPU: {gpu_time:.2f} segundos")
else:
    print(" GPU no detectada")
    # Entrenamiento en GPU si está disponible
if tf.config.list_physical_devices("GPU"):
    with tf.device("/GPU:0"):
        model_gpu = build_model(input_shape=(100,), vocab_size=5000, num_classes=5)
        model_gpu.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        start_gpu = time.time()
        model_gpu.fit(x_train, y_train, epochs=3, batch_size=1024, validation_data=(x_val, y_val), verbose=1)
        gpu_time = time.time() - start_gpu
        print(f" Tiempo total en GPU: {gpu_time:.2f} segundos")
else:
    print("️ GPU no detectada")
    # Entrenamiento en GPU si está disponible
if tf.config.list_physical_devices("GPU"):
    with tf.device("/GPU:0"):
        model_gpu = build_model(input_shape=(100,), vocab_size=5000, num_classes=5)
        model_gpu.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        start_gpu = time.time()
        model_gpu.fit(x_train, y_train, epochs=3, batch_size=1024, validation_data=(x_val, y_val), verbose=1)
        gpu_time = time.time() - start_gpu
        print(f" Tiempo total en GPU: {gpu_time:.2f} segundos")
else:
    print(" GPU no detectada")
    # Entrenamiento en GPU si está disponible
if tf.config.list_physical_devices("GPU"):
    with tf.device("/GPU:0"):
        model_gpu = build_model(input_shape=(100,), vocab_size=5000, num_classes=5)
        model_gpu.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        start_gpu = time.time()
        model_gpu.fit(x_train, y_train, epochs=3, batch_size=1024, validation_data=(x_val, y_val), verbose=1)
        gpu_time = time.time() - start_gpu
        print(f"Tiempo total en GPU: {gpu_time:.2f} segundos")
else:
    print(" GPU no detectada")