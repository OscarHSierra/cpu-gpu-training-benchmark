import tensorflow as tf
import time
import numpy as np
import random
import os

# Fijar semilla para reproducibilidad
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

# (Opcional) Control de memoria dinámica en GPU para evitar asignación total de memoria
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        pass  # En caso de que no sea soportado en tu entorno

# Función de entrenamiento
def run_training(device_name):
    print(f"\nEntrenando en: {device_name}")
    with tf.device(device_name):
        # Cargar dataset de Shakespeare
        path = tf.keras.utils.get_file('shakespeare.txt',
            'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
        text = open(path, 'rb').read().decode(encoding='utf-8')
        vocab = sorted(set(text))
        char2idx = {u: i for i, u in enumerate(vocab)}
        idx2char = np.array(vocab)
        text_as_int = np.array([char2idx[c] for c in text])

        # Preparar dataset
        seq_length = 100
        char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
        sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)

        def split_input_target(chunk):
            input_text = chunk[:-1]
            target_text = chunk[1:]
            return input_text, target_text

        dataset = sequences.map(split_input_target)
        BATCH_SIZE = 64
        BUFFER_SIZE = 10000
        dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

        # Modelo RNN profundo
        vocab_size = len(vocab)
        embedding_dim = 256
        rnn_units = 1024

        def build_deep_rnn_model(vocab_size, embedding_dim, rnn_units):
            model = tf.keras.Sequential([
                tf.keras.layers.Embedding(vocab_size, embedding_dim),
                tf.keras.layers.SimpleRNN(rnn_units, return_sequences=True),
                tf.keras.layers.SimpleRNN(rnn_units, return_sequences=True),
                tf.keras.layers.SimpleRNN(rnn_units, return_sequences=True),
                tf.keras.layers.Dense(vocab_size)
            ])
            return model

        model = build_deep_rnn_model(vocab_size, embedding_dim, rnn_units)
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

        # Entrenamiento
        start_time = time.time()
        model.fit(dataset, epochs=1)
        elapsed = time.time() - start_time
        print(f" Tiempo de entrenamiento en {device_name}: {elapsed:.2f} segundos")
        return elapsed

# Entrenamiento en CPU
# cpu_time = run_training("/CPU:0")
# cpu_time = run_training("/CPU:0")
# cpu_time = run_training("/CPU:0")
# cpu_time = run_training("/CPU:0")
# cpu_time = run_training("/CPU:0")
# Entrenamiento en GPU (si disponible)
if tf.config.list_physical_devices('GPU'):
    gpu_time = run_training("/GPU:0")
    gpu_time = run_training("/GPU:0")
    gpu_time = run_training("/GPU:0")
    gpu_time = run_training("/GPU:0")
    gpu_time = run_training("/GPU:0")
else:
    print("\n GPU no disponible.")
