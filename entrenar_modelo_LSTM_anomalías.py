import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, text
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import root_mean_squared_error, mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# --- 1. CONFIGURACIÓN ---
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

df = df.rename(columns={"valor_real": "demanda_real", 
                        "valor_previsto": "demanda_prevista", 
                        "valor_programado": "demanda_programada"})

# --- 3. ESCALADO DE DATOS (Múltiples Características) ---
features_to_scale = ['demanda_real', 'demanda_prevista', 'demanda_programada']
scaler = MinMaxScaler(feature_range=(0, 1))
data_escalada = scaler.fit_transform(df[features_to_scale])

# --- 4. INGENIERÍA DE CARACTERÍSTICAS DE SECUENCIA ---
LOOK_BACK = 24  # 2 horas de datos para predecir el siguiente punto
def create_sequences_multivariate(data, look_back):
    X, Y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back), :])
        Y.append(data[i + look_back, 0])
    return np.array(X), np.array(Y)

train_size = int(len(data_escalada) * 0.8)
train_data = data_escalada[:train_size]
test_data = data_escalada[train_size:]

X_train, y_train = create_sequences_multivariate(train_data, LOOK_BACK)
X_test, y_test = create_sequences_multivariate(test_data, LOOK_BACK)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2]))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))

# --- 5. ENTRENAMIENTO Y PREDICCIÓN (Modelo LSTM omitido por brevedad en este bloque) ---
# Se asume que el entrenamiento se ejecuta y el modelo es cargado o re-entrenado
num_features = X_train.shape[2]
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(LOOK_BACK, num_features)))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)], verbose=0)
# --- 5.1 PREDICCIÓN Y TRANSFORMACIÓN ---
y_pred_escalada = model.predict(X_test, verbose=0)

dummy_array_test = np.zeros((len(y_test), len(features_to_scale)))
dummy_array_test[:, 0] = y_test
y_test_original = scaler.inverse_transform(dummy_array_test)[:, 0]

dummy_array_pred = np.zeros((len(y_pred_escalada), len(features_to_scale)))
dummy_array_pred[:, 0] = y_pred_escalada.flatten()
y_pred_original = scaler.inverse_transform(dummy_array_pred)[:, 0]

# Cálculo de métricas
rmse = root_mean_squared_error(y_test_original, y_pred_original, squared=False)
mape = mean_absolute_percentage_error(y_test_original, y_pred_original) * 100
indices_test = df.index[train_size + LOOK_BACK:]

# --- 6. ANÁLISIS DE RESIDUOS Y DETECCIÓN DE ANOMALÍAS (NUEVO) ---

# Crear un DataFrame de resultados con los valores originales
resultados_test = pd.DataFrame({
    'real': y_test_original, 
    'prediccion': y_pred_original
}, index=indices_test)

# 6.1. Calcular el error de predicción (Residuos)
resultados_test['residuos'] = resultados_test['real'] - resultados_test['prediccion']

# 6.2. Definir el Umbral (3.5 * Desviación Estándar)
sigma = resultados_test['residuos'].std()
THRESHOLD = 3.5 * sigma 

print(f"\nError (RMSE) del modelo LSTM en los datos de test: {rmse:.2f} MW")
print(f"Umbral de Anomalía (3.5-sigma): {THRESHOLD:.2f} MW")

# 6.3. Etiquetar Anomalías
resultados_test['anomalia'] = np.where(
    resultados_test['residuos'].abs() > THRESHOLD, 
    1, # Anomalía Detectada
    0  # Comportamiento Normal
)

anomalias_detectadas = resultados_test[resultados_test['anomalia'] == 1]
print(f"Total de anomalías detectadas: {len(anomalias_detectadas)}")

# --- 7. VISUALIZACIÓN CON ANOMALÍAS RESALTADAS ---

plt.figure(figsize=(16, 8))
plt.style.use('seaborn-v0_8-whitegrid')

plt.plot(resultados_test.index, resultados_test['real'], 
         label='Demanda Real (MW)', 
         color='steelblue', 
         linewidth=1.5)

plt.plot(resultados_test.index, resultados_test['prediccion'], 
         label='Predicción LSTM (MW)', 
         color='forestgreen', 
         linewidth=2.5, 
         alpha=0.8)

# Resaltar anomalías con puntos rojos
if not anomalias_detectadas.empty:
    plt.scatter(anomalias_detectadas.index, anomalias_detectadas['real'], 
                label='Anomalía Detectada', 
                color='red', 
                s=70, 
                zorder=5)

plt.title(f'Detección de Anomalías con LSTM | RMSE: {rmse:.2f} MW', fontsize=18)
plt.xlabel("Fecha y Hora", fontsize=14)
plt.ylabel("Potencia (MW)", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()