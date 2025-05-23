import os
import logging
from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy import stats

# Configuración del registro de mensajes
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def cargar_datos(ruta_archivo: str) -> pd.DataFrame:
    """
    Cargar los datos CSV, omitiendo las dos primeras filas.

    Args:
        ruta_archivo: Ruta al archivo de datos.

    Devuelve:
        DataFrame con las columnas ['t', 'x', 'y'].
    """
    return pd.read_csv(ruta_archivo, skiprows=2, header=None, names=['t', 'x', 'y'], delimiter=',')

def calcular_altura_umbral(senal: np.ndarray) -> float:
    """
    Calcular el umbral mínimo de altura para detección de picos.

    Args:
        senal: Arreglo 1D con los valores de la señal.

    Devuelve:
        Umbral de altura como float.
    """
    return 0.5 * (senal.max() - senal.min()) + senal.min()

def detectar_picos(senal: np.ndarray, altura: float, distancia: int) -> np.ndarray:
    """
    Detectar los índices de los picos en la señal.

    Args:
        senal: Arreglo 1D.
        altura: Altura mínima de los picos.
        distancia: Distancia mínima entre picos.

    Devuelve:
        Índices de los picos.
    """
    picos, _ = find_peaks(senal, height=altura, distance=distancia)
    return picos

def calcular_omega_y_periodo(t: np.ndarray, picos: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Calcular la frecuencia angular ω y el período T con sus incertidumbres.

    Args:
        t: Arreglo de tiempos.
        picos: Índices de los picos detectados.

    Devuelve:
        omega, sigma_omega, periodo, sigma_periodo
    """
    if len(picos) < 2:
        return np.nan, np.nan, np.nan, np.nan

    t_picos = t[picos]
    intervalos = np.diff(t_picos)
    n = len(intervalos)

    periodo_medio = intervalos.mean()
    sigma_periodo = intervalos.std(ddof=1) / np.sqrt(n)

    omega = 2 * np.pi / periodo_medio
    sigma_omega = (2 * np.pi / periodo_medio**2) * sigma_periodo

    return omega, sigma_omega, periodo_medio, sigma_periodo

def obtener_amplitud_inicial(senal: np.ndarray, picos: np.ndarray) -> float:
    """
    Obtener la amplitud de oscilación inicial (valor absoluto del primer pico).

    Devuelve:
        Amplitud del primer pico o NaN si no hay picos.
    """
    if len(picos) == 0:
        return np.nan
    return abs(senal[picos[0]])

def amplitud_en_el_tiempo(senal: np.ndarray, t: np.ndarray, picos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extraer las amplitudes y sus tiempos a partir de los picos.

    Devuelve:
        tiempos y amplitudes como arrays.
    """
    return t[picos], np.abs(senal[picos])

def graficar_señales(lista_df: List[pd.DataFrame], nombres: List[str], altura: float, distancia: int) -> None:
    """
    Graficar todas las señales con los picos detectados.

    Args:
        lista_df: Lista de DataFrames con 't' y 'x'.
        nombres: Lista de nombres de archivos.
        altura: Umbral de altura para detección de picos.
        distancia: Distancia mínima entre picos.
    """
    for df, nombre in zip(lista_df, nombres):
        t = df['t'].to_numpy()
        senal = df['x'].to_numpy()
        picos = detectar_picos(senal, altura, distancia)
        plt.figure(figsize=(7,3))
        plt.plot(t, senal, label='x(t)')
        plt.plot(t[picos], senal[picos], 'ro', label='Picos')
        plt.xlabel('Tiempo (s)')
        plt.ylabel('Posición x (cm)')
        plt.title(nombre)
        plt.legend()
        plt.grid(True)
        plt.show()

def graficar_omega_vs_longitud(df: pd.DataFrame) -> None:
    marcadores = ['o', 's', '^']
    estilos = ['-', '--', ':']
    plt.figure(figsize=(8,5))
    for (masa, grupo), marcador, estilo in zip(df.groupby('masa'), marcadores, estilos):
        plt.errorbar(grupo['longitud_cm'], grupo['omega'], yerr=grupo['sigma_omega'],
                     fmt=marcador, linestyle=estilo, capsize=3, label=masa)
    plt.xlabel('Longitud (cm)')
    plt.ylabel('Frecuencia angular ω (rad/s)')
    plt.title('ω vs Longitud para diferentes masas')
    plt.legend()
    plt.grid(True)
    plt.show()

def graficar_omega_vs_masa(df: pd.DataFrame) -> None:
    plt.figure(figsize=(8,5))
    for longitud, grupo in df.groupby('longitud_cm'):
        plt.errorbar(grupo['masa_kg'], grupo['omega'], yerr=grupo['sigma_omega'],
                     fmt='o', capsize=3, label=f'L={longitud} cm')
    plt.xlabel('Masa (kg)')
    plt.ylabel('Frecuencia angular ω (rad/s)')
    plt.title('ω vs Masa para diferentes longitudes')
    plt.legend()
    plt.grid(True)
    plt.show()

def graficar_omega_vs_amplitud(df: pd.DataFrame, masa: str, longitud_cm: float) -> None:
    grupo = df[(df['masa'] == masa) & (df['longitud_cm'] == longitud_cm)]
    plt.errorbar(grupo['amplitud_inicial'], grupo['omega'], yerr=grupo['sigma_omega'], fmt='o', capsize=3)
    plt.xlabel('Amplitud inicial (cm)')
    plt.ylabel('Frecuencia angular ω (rad/s)')
    plt.title(f'ω vs Amplitud inicial para masa={masa}, longitud={longitud_cm} cm')
    plt.grid(True)
    plt.show()

def graficar_amplitud_vs_tiempo(t: np.ndarray, senal: np.ndarray, picos: np.ndarray, nombre_archivo: str, carpeta='graficos'):
    """
    Graficar amplitud vs tiempo usando los picos de la señal.
    """
    tiempos, amplitudes = amplitud_en_el_tiempo(senal, t, picos)
    plt.figure(figsize=(8, 4))
    plt.plot(tiempos, amplitudes, 'o-', label='Amplitud |x|')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Amplitud (cm)')
    plt.title(f'Amplitud vs Tiempo - {nombre_archivo}')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(carpeta, f'{nombre_archivo}_amplitud_tiempo.png'), dpi=300)
    plt.close()

def agregar_omega_teorica(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega ω_teorica y el desvío relativo al DataFrame.
    """
    df = df.copy()
    g = 9.81  # m/s²
    df['longitud_m'] = df['longitud_cm'] / 100
    df['omega_teorica'] = np.sqrt(g / df['longitud_m'])
    df['delta_omega'] = df['omega'] - df['omega_teorica']
    df['desvio_relativo'] = np.abs(df['delta_omega']) / df['omega_teorica']
    return df

def regresion_omega_vs_masa(df: pd.DataFrame):
    """
    Ajuste ω vs masa para demostrar independencia.
    """
    for longitud, grupo in df.groupby('longitud_cm'):
        pendiente, intercepto, r, _, stderr = stats.linregress(grupo['masa_kg'], grupo['omega'])
        print(f"[Longitud {longitud} cm] ω = {pendiente:.3f} * m + {intercepto:.3f} | r = {r:.3f} | error estándar = {stderr:.3e}")

def regresion_T2_vs_L(df: pd.DataFrame) -> pd.DataFrame:
    g_estimados = []
    for masa, grupo in df.groupby('masa'):
        L_m = grupo['longitud_cm'].to_numpy() / 100
        T2 = grupo['periodo'].to_numpy()**2
        sigma_T = grupo['sigma_periodo'].to_numpy()
        sigma_T2 = 2 * grupo['periodo'] * sigma_T
        pendiente, intercepto, _, _, error_pend = stats.linregress(L_m, T2)
        g_estimado = 4 * np.pi**2 / pendiente
        sigma_g = (4 * np.pi**2 / pendiente**2) * error_pend

        g_estimados.append({
            'masa': masa,
            'g_estimado': g_estimado,
            'sigma_g': sigma_g,
            'pendiente': pendiente,
            'error_pendiente': error_pend,
            'intercepto': intercepto
        })

        plt.errorbar(L_m, T2, yerr=sigma_T2, fmt='o', capsize=3, label='Datos')
        x_fit = np.linspace(L_m.min(), L_m.max(), 100)
        plt.plot(x_fit, pendiente*x_fit + intercepto, 'r-', label=f'Ajuste: pendiente={pendiente:.3f}')
        plt.xlabel('Longitud (m)')
        plt.ylabel('$T^2$ (s²)')
        plt.title(f'Regresión T² vs L para masa {masa}\n g estimada = {g_estimado:.3f} ± {sigma_g:.3f} m/s²')
        plt.legend()
        plt.grid(True)
        plt.show()

    return pd.DataFrame(g_estimados)

def exportar_resultados(df: pd.DataFrame, df_g: pd.DataFrame) -> None:
    df.to_csv('resultados_pendulo_con_incertidumbres_y_teoricos.csv', index=False)
    df_g.to_csv('estimacion_g_por_masa.csv', index=False)
    logging.info("Se guardaron los resultados con comparación ω teórica.")

# --- FUNCIÓN PRINCIPAL ---
def main(carpeta_datos: str, distancia: int = 15) -> None:
    dicc_masa_g = {'Madera': 5.1, 'Aluminio': 22.1, 'Bronce': 72.6}
    dicc_masa_kg = {k: v/1000 for k, v in dicc_masa_g.items()}

    os.makedirs('graficos', exist_ok=True)

    archivos = sorted([f for f in os.listdir(carpeta_datos) if f.endswith('.txt')])
    dfs = [cargar_datos(os.path.join(carpeta_datos, f)) for f in archivos]

    resultados = []

    for df, nombre in zip(dfs, archivos):
        senal = df['x'].to_numpy()
        t = df['t'].to_numpy()
        altura = calcular_altura_umbral(senal)
        picos = detectar_picos(senal, altura, distancia)
        omega, sigma_omega, periodo, sigma_periodo = calcular_omega_y_periodo(t, picos)
        if np.isnan(omega):
            logging.warning(f"No se detectaron suficientes picos en {nombre}, se omite.")
            continue

        partes = nombre.replace('.txt', '').split('_')
        longitud_cm = float(partes[0].replace('cm', ''))
        masa = partes[1]
        masa_kg = dicc_masa_kg.get(masa, np.nan)

        amplitud_inicial = obtener_amplitud_inicial(senal, picos)

        resultados.append({
            'archivo': nombre,
            'masa': masa,
            'masa_kg': masa_kg,
            'longitud_cm': longitud_cm,
            'amplitud_inicial': amplitud_inicial,
            'omega': omega,
            'sigma_omega': sigma_omega,
            'periodo': periodo,
            'sigma_periodo': sigma_periodo
        })
        graficar_señales([df], [nombre], altura, distancia)
        graficar_amplitud_vs_tiempo(t, senal, picos, nombre)
        


    df_resultados = pd.DataFrame(resultados)
    logging.info(f"Se procesaron correctamente {len(df_resultados)} archivos.")

    df_resultados = agregar_omega_teorica(df_resultados)

    graficar_omega_vs_longitud(df_resultados)
    graficar_omega_vs_masa(df_resultados)

    if not df_resultados.empty:
        ejemplo_masa = df_resultados['masa'].iloc[0]
        ejemplo_longitud = df_resultados['longitud_cm'].iloc[0]
        graficar_omega_vs_amplitud(df_resultados, ejemplo_masa, ejemplo_longitud)

    df_g = regresion_T2_vs_L(df_resultados)
    regresion_omega_vs_masa(df_resultados)

    exportar_resultados(df_resultados, df_g)

# --- EJECUCIÓN ---
if __name__ == '__main__':
    carpeta_datos = r'C:\Users\morph\Desktop\UDESA\fisica\Informes_Fisica\2do_Informe\CSVs'  # Cambiar si es necesario
    main(carpeta_datos)
