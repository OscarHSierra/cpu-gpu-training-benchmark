import xgboost as xgb
import numpy as np
import random
import time
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -------------------------------
# Fijar semilla para reproducibilidad
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
xgb.set_config(verbosity=1)  # No es la semilla, pero limpia logs ruidosos
# -------------------------------

# Crear un dataset grande
X, y = make_classification(n_samples=200000, n_features=1000, n_informative=75, 
                           n_classes=5, random_state=SEED)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

# Parámetros comunes
params = {
    'objective': 'multi:softmax',
    'num_class': 5,
    'max_depth': 6,
    'eta': 0.1,
    'eval_metric': 'mlogloss',
    'seed': SEED  # Semilla para XGBoost
}

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# ----- CPU -----
params_cpu = params.copy()
params_cpu['tree_method'] = 'auto'

print("Entrenando con CPU...")
start_cpu = time.time()
model_cpu = xgb.train(params_cpu, dtrain, num_boost_round=100)
cpu_time = time.time() - start_cpu
preds_cpu = model_cpu.predict(dtest)
acc_cpu = accuracy_score(y_test, preds_cpu)
print(f"Tiempo CPU: {cpu_time:.2f} segundos - Accuracy: {acc_cpu:.4f}")

# ----- GPU -----
params_gpu = params.copy()
params_gpu['tree_method'] = 'gpu_hist'
params_gpu['predictor'] = 'gpu_predictor'

print("\nEntrenando con GPU...")
start_gpu = time.time()
model_gpu = xgb.train(params_gpu, dtrain, num_boost_round=100)
gpu_time = time.time() - start_gpu
preds_gpu = model_gpu.predict(dtest)
acc_gpu = accuracy_score(y_test, preds_gpu)
print(f" Tiempo GPU: {gpu_time:.2f} segundos - Accuracy: {acc_gpu:.4f}")
