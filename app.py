from itertools import groupby
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

st.title("RESUMEN DE SIMULACIONES - MÉTRICAS AVANZADAS")

# Datos iniciales
relacion = st.sidebar.number_input("Relación R/B", value=1.0)
probabilidad = st.sidebar.number_input("Probabilidad (%)", value=55.0)
riesgo = st.sidebar.number_input("Riesgo por operación", value=200.0)
canttrades = st.sidebar.number_input("Número de Trades", min_value=100, value=1000)
targetpruebafondeo = st.sidebar.number_input("Target Prueba Fondeo", value=1750.0)
perdidamaxima = st.sidebar.number_input("Pérdida Máxima", value=-1500.0)
num_pruebas = st.sidebar.number_input("Número de Pruebas", min_value=1, value=5)

# Calcular probabilidad negativa
probpositivos = probabilidad / 100
probnegativos = 1 - probpositivos

# Almacenar resultados de múltiples simulaciones
resultados_pruebas = []
estadisticas_resumen = {
    "Pruebas Pasadas": 0,
    "Pruebas Quemadas": 0,  # Nueva métrica
    "Max Drawdown": [],
    "Ganancias Totales": [],
    "Pérdidas Totales": [],
    "Max Positivos Consecutivos": [],
    "Max Negativos Consecutivos": [],
    "R2": []
}

for i in range(num_pruebas):
    # Simulación de resultados de trades
    df = pd.DataFrame({
        'Resultado': pd.Series([1, -1]).sample(
            canttrades,
            replace=True,
            weights=[probpositivos, probnegativos]
        ).values
    })

    # Función para calcular el Net Profit
    def calcular_netprofit(row):
        if row['Resultado'] == -1:
            return riesgo * -1
        else:
            return riesgo * relacion

    # Aplicar cálculo al DataFrame
    df['Net Profit'] = df.apply(calcular_netprofit, axis=1)
    df['Net Profit Cum'] = df['Net Profit'].cumsum()

    # Calcular Drawdown
    df['Max Cum Profit'] = df['Net Profit Cum'].cummax()
    df['Drawdown'] = df['Net Profit Cum'] - df['Max Cum Profit']

    # Evaluar máximos consecutivos
    max_pos = max((sum(1 for _ in g) for k, g in groupby(df['Resultado']) if k == 1), default=0)
    max_neg = max((sum(1 for _ in g) for k, g in groupby(df['Resultado']) if k == -1), default=0)

    # Calcular R²
    x = np.arange(len(df)).reshape(-1, 1)  # Índices como variable independiente
    y = df['Net Profit Cum'].values.reshape(-1, 1)
    model = LinearRegression().fit(x, y)
    r2 = model.score(x, y)

    # Evaluar pruebas fondeo
    suma = 0
    prueba_pasada = False
    prueba_quemada = False  # Nueva variable para pruebas quemadas
    for profit in df['Net Profit']:
        suma += profit
        if suma >= targetpruebafondeo:
            prueba_pasada = True
            break
        if suma <= perdidamaxima:
            prueba_quemada = True  # Marcar como quemada
            break

    # Actualizar estadísticas de resumen
    estadisticas_resumen["Pruebas Pasadas"] += int(prueba_pasada)
    if prueba_quemada:
        estadisticas_resumen["Pruebas Quemadas"] += 1  # Contabilizar pruebas quemadas
    estadisticas_resumen["Max Drawdown"].append(df['Drawdown'].min())
    estadisticas_resumen["Ganancias Totales"].append(df[df['Net Profit'] > 0]['Net Profit'].sum())
    estadisticas_resumen["Pérdidas Totales"].append(df[df['Net Profit'] < 0]['Net Profit'].sum())
    estadisticas_resumen["Max Positivos Consecutivos"].append(max_pos)
    estadisticas_resumen["Max Negativos Consecutivos"].append(max_neg)
    estadisticas_resumen["R2"].append(r2)

    # Almacenar resultados individuales
    resultados_pruebas.append(df)

# Resumen final
total_pruebas = len(resultados_pruebas)
probabilidad_pasar = estadisticas_resumen["Pruebas Pasadas"] / total_pruebas * 100
max_drawdown = min(estadisticas_resumen["Max Drawdown"])
ganancias_totales = sum(estadisticas_resumen["Ganancias Totales"])
perdidas_totales = sum(estadisticas_resumen["Pérdidas Totales"])
profit_factor = ganancias_totales / abs(perdidas_totales) if perdidas_totales != 0 else float('inf')
max_pos_consec = max(estadisticas_resumen["Max Positivos Consecutivos"])
max_neg_consec = max(estadisticas_resumen["Max Negativos Consecutivos"])
r2_promedio = np.mean(estadisticas_resumen["R2"])

# Visualización del resumen
st.header("Resumen de Simulaciones")
st.write(f"**Cantidad de Pruebas Pasadas:** {estadisticas_resumen['Pruebas Pasadas']} de {total_pruebas}")
st.write(f"**Cantidad de Pruebas Quemadas:** {estadisticas_resumen['Pruebas Quemadas']} de {total_pruebas}")
st.write(f"**Probabilidad de Pasar las Pruebas:** {probabilidad_pasar:.2f}%")
st.write(f"**Máximo Drawdown:** {max_drawdown:.2f}")
st.write(f"**Profit Factor:** {profit_factor:.2f}")
st.write(f"**Ganancias Totales:** ${ganancias_totales:.2f}")
st.write(f"**Pérdidas Totales:** ${perdidas_totales:.2f}")
st.write(f"**Máximo Trades Positivos Consecutivos:** {max_pos_consec}")
st.write(f"**Máximo Trades Negativos Consecutivos:** {max_neg_consec}")
st.write(f"**R² Promedio de las Simulaciones:** {r2_promedio:.4f}")

# Mostrar resultados individuales
st.header("Resultados Individuales")
for idx, df in enumerate(resultados_pruebas):
    st.subheader(f"Simulación {idx + 1}")
    st.line_chart(df, y="Net Profit Cum")
