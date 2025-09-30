import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, text
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import root_mean_squared_error, mean_absolute_percentage_error, mean_absolute_error  # <<< añadido
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import pyodbc  # (asegúrate de tenerlo instalado)

# --- 1. CONEXIÓN A LA BASE DE DATOS ---
SERVER   = "udcserver2025.database.windows.net"
DATABASE = "grupo_1"
USER     = "ugrupo1"
PASSWORD = "HK9WXIJaBp2Q97haePdY"

ENGINE_URL = f"mssql+pyodbc://{USER}:{PASSWORD}@{SERVER}:1433/{DATABASE}?driver=ODBC+Driver+18+for+SQL+Server&Encrypt=yes&TrustServerCertificate=no"
engine = create_engine(ENGINE_URL, pool_pre_ping=True)

# --- 2. CARGAR Y PREPARAR DATOS (Tabla 'demanda_total') ---
sql = text("""
    SELECT
        fecha,
        valor_real,
        valor_previsto,
        valor_programado
    FROM dbo.demanda_total
    ORDER BY fecha;
""")
df = pd.read_sql(sql, engine, parse_dates=["fecha"]).dropna()
df = df.set_index("fecha").sort_index()

# 2.1. Normalización de nombres de columnas
df = df.rename(columns={"valor_real": "demanda_real", 
                        "valor_previsto": "demanda_prevista", 
                        "valor_programado": "demanda_programada"})

# --- 3. ESCALADO DE DATOS (Múltiples Características) ---
features_to_scale = ['demanda_real', 'demanda_prevista', 'demanda_programada']
scaler = MinMaxScaler(feature_range=(0, 1))
data_escalada = scaler.fit_transform(df[features_to_scale])

# --- 4. DIVISIÓN DEL CONJUNTO DE DATOS ---
train_size = int(len(data_escalada) * 0.8)
train_data = data_escalada[:train_size]
test_data = data_escalada[train_size:]

# --- 5. INGENIERÍA DE CARACTERÍSTICAS DE SECUENCIA ---
LOOK_BACK = 24  # 2 horas de datos para predecir el siguiente punto

def create_sequences_multivariate(data, look_back):
    X, Y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back), :])   # ventana con 3 features
        Y.append(data[i + look_back, 0])       # objetivo: demanda_real siguiente
    return np.array(X), np.array(Y)

X_train, y_train = create_sequences_multivariate(train_data, LOOK_BACK)
X_test, y_test = create_sequences_multivariate(test_data, LOOK_BACK)

print(f"Forma de X_train (LSTM 3D): {X_train.shape}")
print(f"Datos de entrenamiento: {len(y_train)} | Datos de test: {len(y_test)}")

# --- 6. DEFINICIÓN DEL MODELO LSTM ---
print("\nDefiniendo y entrenando el modelo LSTM...")
num_features = X_train.shape[2]

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(LOOK_BACK, num_features)))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
history = model.fit(
    X_train, y_train, 
    epochs=100, 
    batch_size=32, 
    validation_data=(X_test, y_test),
    callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)],
    verbose=0
)

# --- 7. PREDICCIÓN Y MÉTRICAS ---
y_pred_escalada = model.predict(X_test, verbose=0)

# Invertir el escalado
dummy_array_test = np.zeros((len(y_test), len(features_to_scale)))
dummy_array_test[:, 0] = y_test
y_test_original = scaler.inverse_transform(dummy_array_test)[:, 0]

dummy_array_pred = np.zeros((len(y_pred_escalada), len(features_to_scale)))
dummy_array_pred[:, 0] = y_pred_escalada.flatten()
y_pred_original = scaler.inverse_transform(dummy_array_pred)[:, 0]


rmse = root_mean_squared_error(y_test_original, y_pred_original)
mape = mean_absolute_percentage_error(y_test_original, y_pred_original) * 100
mae  = mean_absolute_error(y_test_original, y_pred_original)  # <<< añadido

# Métricas normalizadas (en %)
nrmse_mean  = rmse / np.mean(y_test_original) * 100          # <<< añadido
nrmse_range = rmse / (np.max(y_test_original) - np.min(y_test_original)) * 100  # <<< añadido

print(f"\nError (RMSE) del modelo LSTM en test: {rmse:.2f} MW")
print(f"Error (MAE)  del modelo LSTM en test: {mae:.2f} MW")        # <<< añadido
print(f"Error (MAPE) del modelo LSTM en test: {mape:.2f}%")
print(f"nRMSE (sobre media): {nrmse_mean:.2f}%")                    # <<< añadido
print(f"nRMSE (sobre rango): {nrmse_range:.2f}%")                   # <<< añadido

# Diagnóstico de error máximo (valor y timestamp)  <<< añadido
indices_test = df.index[train_size + LOOK_BACK:]
err = y_pred_original - y_test_original
imax = int(np.argmax(np.abs(err)))
print(f"Error máximo: {err[imax]:.2f} MW en {indices_test[imax]}")

# --- 8. VISUALIZACIÓN ---
resultados_test = pd.DataFrame({
    'real': y_test_original, 
    'prediccion': y_pred_original
}, index=indices_test)

plt.figure(figsize=(16, 8))
plt.style.use('seaborn-v0_8-whitegrid')
plt.plot(resultados_test.index, resultados_test['real'], label='Demanda Real (MW)', linewidth=1.5)
plt.plot(resultados_test.index, resultados_test['prediccion'], label='Predicción LSTM (MW)', linewidth=2.5, alpha=0.8)

plt.title(
    f'Predicción de Demanda con LSTM | RMSE: {rmse:.2f} MW | MAE: {mae:.2f} MW | '
    f'MAPE: {mape:.2f}% | nRMSE(media): {nrmse_mean:.2f}%',
    fontsize=18
)  # <<< título enriquecido

plt.xlabel("Fecha y Hora", fontsize=14)
plt.ylabel("Potencia (MW)", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


model.save("modelo_lstm_multivariate.keras")


