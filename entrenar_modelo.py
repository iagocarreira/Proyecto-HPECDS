import pandas as pd
import lightgbm as lgb
from sqlalchemy import create_engine
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt


# --- 1. CONFIGURACIÓN ---
SERVER   = "udcserver2025.database.windows.net"
DATABASE = "grupo_1"
USER     = "ugrupo1"
PASSWORD = "HK9WXIJaBp2Q97haePdY"

ENGINE_URL = f"mssql+pyodbc://{USER}:{PASSWORD}@{SERVER}:1433/{DATABASE}?driver=ODBC+Driver+18+for+SQL+Server&Encrypt=yes&TrustServerCertificate=no"
engine = create_engine(ENGINE_URL, pool_pre_ping=True)

# --- 2. CARGAR Y PREPARAR DATOS ---
query = "SELECT fecha, valor FROM dbo.demanda ORDER BY fecha"
df = pd.read_sql(query, engine, index_col='fecha', parse_dates=True)
df = df.sort_index().dropna()

# --- 3. INGENIERÍA DE CARACTERÍSTICAS  ---

def crear_features_avanzadas(dataframe):
    df_copy = dataframe.copy()
    
    # 3.1. Features temporales 
    df_copy['hora'] = df_copy.index.hour
    df_copy['dia_semana'] = df_copy.index.dayofweek
    df_copy['dia_mes'] = df_copy.index.day
    df_copy['mes'] = df_copy.index.month
    
    # 3.2. Lag Features
    df_copy['lag_5min'] = df_copy['valor'].shift(1) 
    df_copy['lag_1h'] = df_copy['valor'].shift(12)  
    df_copy['lag_24h'] = df_copy['valor'].shift(288)
    
    return df_copy.dropna()

df_features = crear_features_avanzadas(df)

# --- 4. DIVISIÓN DEL CONJUNTO DE DATOS  ---
train_size = int(len(df_features) * 0.8)
train_df = df_features.iloc[:train_size]
test_df = df_features.iloc[train_size:]

X_train = train_df.drop('valor', axis=1)
y_train = train_df['valor']
X_test = test_df.drop('valor', axis=1)
y_test = test_df['valor']

# --- 5. ENTRENAMIENTO CON LIGHTGBM ---
print("Entrenando el modelo LightGBM...")

lgb_reg = lgb.LGBMRegressor(
    objective='regression', 
    metric='rmse', 
    n_estimators=1000, 
    learning_rate=0.05, 
    num_leaves=31, 
    n_jobs=-1
)

lgb_reg.fit(X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.early_stopping(100, verbose=False)])

# --- 6. PREDICCIÓN Y CÁLCULO DE MÉTRICAS ---
y_pred = lgb_reg.predict(X_test)

# Cálculo de RMSE y MAPE
rmse = mean_squared_error(y_test, y_pred, squared=False)
mape = mean_absolute_percentage_error(y_test, y_pred) * 100

print(f"\nError (RMSE) del modelo en los datos de test: {rmse:.2f} MW")
print(f"Error (MAPE) del modelo en los datos de test: {mape:.2f}%")

# 7. CREAR DATAFRAME DE RESULTADOS
resultados_test = pd.DataFrame({
    'real': y_test, 
    'prediccion': y_pred
}, index=y_test.index)


# --- 8. GENERAR GRÁFICO ---
plt.figure(figsize=(16, 8))
plt.style.use('seaborn-v0_8-whitegrid')

# Curva de Demanda Real
plt.plot(resultados_test.index, resultados_test['real'], 
         label='Demanda Real (MW)', 
         color='steelblue', 
         linewidth=1.5)

# Curva de Predicción del Modelo
plt.plot(resultados_test.index, resultados_test['prediccion'], 
         label='Predicción LightGBM (MW)', 
         color='darkorange', 
         linewidth=2.5, 
         alpha=0.8)

# Formato del Gráfico
plt.title(f'Predicción de Demanda vs. Realidad | RMSE: {rmse:.2f} MW | MAPE: {mape:.2f}%', fontsize=18)
plt.xlabel("Fecha y Hora", fontsize=14)
plt.ylabel("Potencia (MW)", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# Al final de tu script LightGBM:
lgb_reg.booster_.save_model("modelo_lgbm_multivariate.txt")