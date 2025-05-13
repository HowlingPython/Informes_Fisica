import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.odr import ODR, Model, RealData
from math import pi, sqrt

# --- Errores instrumentales ---
delta_M = 0.01  # g
delta_D = 0.05  # cm

# --- Funciones Auxiliares ---

def ajuste_lineal(x, y, sigma_x, sigma_y):
    """
    Realiza un ajuste lineal considerando errores solo en y, y calcula la matriz de covarianza.

    Fórmulas:
        y = a·x + b

    Parámetros:
        x (array): Valores independientes (x).
        y (array): Valores dependientes (y).
        sigma_x (float or array): Incertidumbre en x (puede ser 0).
        sigma_y (float or array): Incertidumbre en y.

    Retorna:
        tuple: (pendiente a, ordenada b, matriz de covarianza 2x2)
    """

    X, Y = np.mean(x), np.mean(y)
    X2, XY = np.mean(x**2), np.mean(x*y)
    N = x.size
    dX2 = X2 - X**2

    a = (XY - X * Y) / dX2
    b = (Y * X2 - XY * X) / dX2

    da_dx = ((y - Y) * dX2 - 2 * (x - X) * (XY - X * Y)) / (N * dX2**2)
    da_dy = (x - X) / (N * dX2)
    db_dx = ((2 * x * Y - y * X - XY) * dX2 - 2 * (x - X) * (Y * X2 - XY * X)) / (N * dX2**2)
    db_dy = (X2 - x * X) / (N * dX2)

    var_a = np.sum(da_dx**2 * sigma_x**2 + da_dy**2 * sigma_y**2)
    var_b = np.sum(db_dx**2 * sigma_x**2 + db_dy**2 * sigma_y**2)
    cov_ab = np.sum(da_dx * db_dx * sigma_x**2 + da_dy * db_dy * sigma_y**2)

    return a, b, np.array([[var_a, cov_ab], [cov_ab, var_b]])

def ajuste_odr(x, y, sx, sy):
    """
    Realiza un ajuste lineal considerando incertidumbre en ambas variables (x e y), utilizando ODR.

    Fórmulas:
        y = a·x + b

    Parámetros:
        x (array): Valores independientes.
        y (array): Valores dependientes.
        sx (array): Incertidumbres en x.
        sy (array): Incertidumbres en y.

    Retorna:
        tuple: (pendiente a, incertidumbre de a, ordenada b, incertidumbre de b)
    """
    def f(B, x):
        return B[0] * x + B[1]

    linear_model = Model(f)
    data = RealData(x, y, sx=sx, sy=sy)
    odr = ODR(data, linear_model, beta0=[1.0, 0.0])
    output = odr.run()
    
    a, b = output.beta
    da, db = output.sd_beta
    return a, da, b, db

def lineal_ajustada(x, a, b, sigma_x, cov):
    """
    Evalúa una recta ajustada y propaga la incertidumbre en y considerando el error en x.

    Fórmulas:
        y = a·x + b
        δy = sqrt((a·δx)² + x²·Var(a) + Var(b) + 2·x·Cov(a,b))

    Parámetros:
        x (array): Valores en el eje x.
        a (float): Pendiente de la recta.
        b (float): Ordenada al origen.
        sigma_x (float or array): Incertidumbre en x.
        cov (2x2 array): Matriz de covarianza del ajuste.

    Retorna:
        tuple: (valores y ajustados, incertidumbres en y)
    """
    y = a * x + b
    sigma_y = np.sqrt(a**2 * sigma_x**2 + x**2 * cov[0, 0] + cov[1, 1] + 2 * x * cov[0, 1])
    return y, sigma_y

def avg_and_error(data, error_inst):
    """
    Calcula el promedio de una serie de mediciones y su incertidumbre total,
    combinando el error estadístico (desviación estándar de la media) y el error instrumental.

    Fórmulas:
        μ = promedio(data)
        δμ = sqrt((δ_inst)² + (σ / sqrt(n))²)

    Parámetros:
        data (list or array): Mediciones realizadas de una misma magnitud.
        error_inst (float): Incertidumbre instrumental (constante para todas las mediciones).

    Retorna:
        tuple: (valor promedio, incertidumbre total)
    """
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)  
    err_stat = std / np.sqrt(n)
    err_total = sqrt(error_inst**2 + err_stat**2)
    return mean, err_total

def compute_area(L, delta_L, W, delta_W):
    """
    Calcula el área de un rectángulo y su incertidumbre, a partir de las 
    medidas del largo y del ancho, considerando errores independientes.

    Fórmulas:
        A = L × W                             [cm²]
        δA = sqrt((W × δL)² + (L × δW)²)      [cm²]

    Parámetros:
        L (float): Largo del rectángulo en centímetros.
        delta_L (float): Incertidumbre en la medición del largo (cm).
        W (float): Ancho del rectángulo en centímetros.
        delta_W (float): Incertidumbre en la medición del ancho (cm).

    Retorna:
        tuple: (Área en cm², incertidumbre del área en cm²)
    """
    A = L * W
    delta_A = sqrt((W * delta_L)**2 + (L * delta_W)**2)
    return A, delta_A

def compute_volume(D, delta_D):
    """
    Calcula el volumen de una esfera y su incertidumbre, considerando que el objeto 
    tiene forma esférica y que se mide su diámetro con una incertidumbre asociada.

    Fórmulas:
        V = (4/3) × π × (D / 2)³     [cm³]
        δV = (π × D² / 2) × δD       [cm³]

    Parámetros:
        D (float): Diámetro promedio del objeto en centímetros.
        delta_D (float): Incertidumbre del diámetro en centímetros.

    Retorna:
        tuple: (Volumen en cm³, incertidumbre del volumen en cm³)
    """
    V = (4/3) * pi * (D / 2)**3
    delta_V = (pi * D**2 / 2) * delta_D
    return V, delta_V

def compute_gramaje(M, A, delta_A):
    """
    Calcula el gramaje del papel (masa por unidad de área) y su incertidumbre,
    convirtiendo el resultado de g/cm² a g/m².

    Fórmulas:
        G = (M / A) × 10⁴   [g/m²]
        δG = G × sqrt((δM / M)² + (δA / A)²)

    Parámetros:
        M (float): Masa de la hoja en gramos.
        A (float): Área de la hoja en cm².
        delta_A (float): Incertidumbre del área en cm².

    Retorna:
        tuple: (Gramaje en g/m², incertidumbre del gramaje en g/m²)
    """
    G_cm2 = M / A
    delta_G_cm2 = G_cm2 * sqrt((delta_M / M)**2 + (delta_A / A)**2)
    return G_cm2 * 10000, delta_G_cm2 * 10000

def modelo_potencia(m, A, b):
    """
    Modelo de ley de potencias para ajuste no lineal de la forma: D = A·M^b

    Parámetros:
        m (array): Masa.
        A (float): Coeficiente de la potencia.
        b (float): Exponente.

    Retorna:
        array: Diámetros modelados.
    """
    return A * m**b

def ajustar_con_curve_fit(muestras):
    """
    Ajusta los datos experimentales de diámetro en función de masa usando una ley de potencias.

    Modelo:
        D = A·M^b

    Parámetros:
        muestras (list): Lista de diccionarios con datos experimentales.

    Retorna:
        tuple: (A, δA, b, δb)
            A (float): Coeficiente ajustado.
            δA (float): Incertidumbre de A.
            b (float): Exponente ajustado.
            δb (float): Incertidumbre de b.
    """
    masas = np.array([m["M"] for m in muestras])
    diametros = np.array([avg_and_error(m["diameters"], delta_D)[0] for m in muestras])
    errores_diametro = np.array([avg_and_error(m["diameters"], delta_D)[1] for m in muestras])

    popt, pcov = curve_fit(modelo_potencia, masas, diametros, sigma=errores_diametro, absolute_sigma=True)
    A_fit, b_fit = popt
    err_A, err_b = np.sqrt(np.diag(pcov))
    return A_fit, err_A, b_fit, err_b

def gramaje_por_ajuste_area_masa(muestras):
    """
    Estima el gramaje del papel realizando un ajuste lineal de Masa vs Área: M = G·A.

    Este método reduce el efecto de errores sistemáticos respecto a calcular G_i = M_i / A_i.

    Parámetros:
        muestras (list): Lista de diccionarios con datos experimentales.

    Retorna:
        tuple: (gramaje en g/cm², incertidumbre del gramaje)
    """
    A_vals, dA_vals, M_vals, dM_vals = [], [], [], []
    for m in muestras:
        A, dA = compute_area(m["L"], m["delta_L"], m["W"], m["delta_W"])
        A_vals.append(A)
        dA_vals.append(dA)
        M_vals.append(m["M"])
        dM_vals.append(delta_M)

    A_vals = np.array(A_vals)
    M_vals = np.array(M_vals)
    dA_vals = np.array(dA_vals)
    dM_vals = np.array(dM_vals)

    G, dG, _, _ = ajuste_odr(A_vals, M_vals, dA_vals, dM_vals)
    return G, dG  # en g/cm²

def analizar_muestra(muestra):
    """
    Calcula todas las propiedades físicas relevantes de una muestra: área, volumen, diámetro y gramaje.

    Parámetros:
        muestra (dict): Diccionario con los datos de una muestra.

    Retorna:
        dict: Diccionario con claves:
            - area (float), delta_area (float)
            - diametro (float), delta_diametro (float)
            - volumen (float), delta_volumen (float)
            - gramaje (float), delta_gramaje (float)
    """
    A, dA = compute_area(muestra["L"], muestra["delta_L"], muestra["W"], muestra["delta_W"])
    D, dD = avg_and_error(muestra["diameters"], delta_D)
    V, dV = compute_volume(D, dD)
    G, dG = compute_gramaje(muestra["M"], A, dA)
    return {
        "area": A, "delta_area": dA,
        "diametro": D, "delta_diametro": dD,
        "volumen": V, "delta_volumen": dV,
        "gramaje": G, "delta_gramaje": dG
    }

def preparar_loglog(muestras):
    """
    Convierte los datos de masa y diámetro al espacio logarítmico para ajuste log-log.

    Parámetros:
        muestras (list): Lista de diccionarios con datos experimentales.

    Retorna:
        tuple: (log(masa), log(diametro), error relativo de log(diametro))
    """
    logM, logD, dlogD = [], [], []
    for m in muestras:
        D, dD = avg_and_error(m["diameters"], delta_D)
        logM.append(np.log(m["M"]))
        logD.append(np.log(D))
        dlogD.append(dD / D)
    return np.array(logM), np.array(logD), np.array(dlogD)

# --- Datos ---

liviano = [
    {"sample": "papel 1", "L": 29.8, "delta_L": 0.05, "W": 29.7, "delta_W": 0.05,"M": 7.44, "diameters": [4.5, 5, 4.6, 4.7, 5.1]},
    {"sample": "papel 2", "L": 12.2, "delta_L": 0.05, "W": 12.2, "delta_W": 0.05, "M": 1.29, "diameters": [2.5, 2.1, 2.2, 1.9, 2.6]},
    {"sample": "papel 3", "L": 12.35, "delta_L": 0.05, "W": 12.20, "delta_W": 0.05, "M": 1.31, "diameters": [2.2, 1.9, 2.3, 2.5, 2]},
    {"sample": "papel 4", "L": 5.15, "delta_L": 0.05, "W": 5.15, "delta_W": 0.05, "M": 0.23, "diameters": [1.3, 1.2, 1.6, 1.4, 1.1]},
    {"sample": "papel 5", "L": 5.15, "delta_L": 0.05, "W": 5.15, "delta_W": 0.05, "M": 0.23, "diameters": [1.4, 1.2, 1.1, 1, 0.9]},
    {"sample": "papel 6", "L": 2.05, "delta_L": 0.05, "W": 1.95, "delta_W": 0.05, "M": 0.11, "diameters": [0.6, 0.5, 0.4]},
    {"sample": "papel 7", "L": 1.85, "delta_L": 0.05, "W": 1.90, "delta_W": 0.05, "M": 0.04, "diameters": [0.6, 0.5, 0.4]},
    {"sample": "papel 8", "L": 1.15, "delta_L": 0.05, "W": 1.15, "delta_W": 0.05, "M": 0.03, "diameters": [0.3, 0.4, 0.2]}
]

medio = [
    {'sample': 'papel 1', 'L': 21.15, 'delta_L': 0.05, 'W': 21.5, 'delta_W': 0.05, 'M': 7.5, 'diameters': [5. , 5.2, 4. , 3.5, 4.6]}, 
    {'sample': 'papel 2', 'L': 8.61, 'delta_L': 0.05, 'W': 8.62, 'delta_W': 0.05, 'M': 1.2, 'diameters': [2.8, 2.1, 2.9, 2.5, 2.7]}, 
    {'sample': 'papel 3', 'L': 8.6, 'delta_L': 0.05, 'W': 8.6, 'delta_W': 0.05, 'M': 1.2, 'diameters': [2.1, 2.2, 1.8, 2. , 2.1]}, 
    {'sample': 'papel 4', 'L': 14.83, 'delta_L': 0.05, 'W': 14.9, 'delta_W': 0.05, 'M': 3.67, 'diameters': [3.5, 3.1, 3.4, 3.3, 2.2]}, 
    {'sample': 'papel 5', 'L': 14.83, 'delta_L': 0.05, 'W': 14.8, 'delta_W': 0.05, 'M': 3.6, 'diameters': [3.1, 2.3, 3.4, 3.2, 3.3]}, 
    {'sample': 'papel 6', 'L': 6.15, 'delta_L': 0.05, 'W': 6.12, 'delta_W': 0.05, 'M': 0.6, 'diameters': [2. , 1.8, 1.5, 1.7, 1.6]},
    {'sample': 'papel 7', 'L': 6.2, 'delta_L': 0.05, 'W': 6.1, 'delta_W': 0.05, 'M': 0.6, 'diameters': [2. , 1.6, 1.8, 1.7, 1.5]}, 
    {'sample': 'papel 8', 'L': 6.3, 'delta_L': 0.05, 'W': 6.25, 'delta_W': 0.05, 'M': 0.65, 'diameters': [1.5, 2. , 1.6, 1.7, 1.8]}, 
    {'sample': 'papel 9', 'L': 6.1, 'delta_L': 0.05, 'W': 6.1, 'delta_W': 0.05, 'M': 0.6, 'diameters': [1.8, 1.3, 1.5, 1.6, 1.4]}
]

pesado = [
    {'sample': 'papel 1', 'L': 29.5, 'delta_L': 0.05, 'W': 29.5, 'delta_W': 0.05, 'M': 21.5, 'diameters': [6, 6.4, 5.6, 6.5, 6.2]}, 
    {'sample': 'papel 2', 'L': 12.4, 'delta_L': 0.05, 'W': 12.3, 'delta_W': 0.05, 'M': 4, 'diameters': [3.8, 3.1, 2.9, 3.7, 3.1]}, 
    {'sample': 'papel 3', 'L': 12.3, 'delta_L': 0.05, 'W': 12.2, 'delta_W': 0.05, 'M': 4, 'diameters': [3.6, 3.8, 3.6, 3.7, 4]}, 
    {'sample': 'papel 4', 'L': 4.9, 'delta_L': 0.05, 'W': 4.8, 'delta_W': 0.05, 'M': 0.6, 'diameters': [1.3, 1.3, 1.2, 1.4, 1.4]}, 
    {'sample': 'papel 5', 'L': 4.8, 'delta_L': 0.05, 'W': 4.8, 'delta_W': 0.05, 'M': 0.5, 'diameters': [1.5, 1.4, 1.9, 1.9, 1.5]}, 
    {'sample': 'papel 6', 'L': 2.1, 'delta_L': 0.05, 'W': 2.1, 'delta_W': 0.05, 'M': 0.11, 'diameters': [0.9, 0.8, 0.9, 0.8]}
]

grupos = {"Liviana": liviano, "Media": medio, "Pesada": pesado}
marcadores = {'Liviana': 'o', 'Media': 's', 'Pesada': '^'}
colores = {'Liviana': 'tab:green', 'Media': 'tab:red', 'Pesada': 'tab:blue'}

valores_b, errores_b = [], []
gramajes_prom, errores_gramaje = [], []

# --- Análisis principal ---

plt.figure(figsize=(10, 7))
for nombre_grupo, muestras in grupos.items():
    masas, diametros, errores_diametro = [], [], []
    gramajes, errores_gram = [], []

    print(f"\nHoja: {nombre_grupo}")
    for muestra in muestras:
        r = analizar_muestra(muestra)
        print(f"{muestra['sample']}: Área={r['area']:.2f}±{r['delta_area']:.2f} cm², "
              f"D={r['diametro']:.2f}±{r['delta_diametro']:.2f} cm, "
              f"V={r['volumen']:.2f}±{r['delta_volumen']:.2f} cm³, "
              f"Gramaje={r['gramaje']:.1f}±{r['delta_gramaje']:.1f} g/m²")

        masas.append(muestra["M"])
        diametros.append(r["diametro"])
        errores_diametro.append(r["delta_diametro"])
        gramajes.append(r["gramaje"])
        errores_gram.append(r["delta_gramaje"])

    gramajes_prom.append(np.mean(gramajes))
    errores_gramaje.append(np.sqrt(np.sum(np.array(errores_gram)**2)) / len(errores_gram))

    # Ajuste log-log
    logM, logD, dlogD = preparar_loglog(muestras)
    pendiente, ordenada, cov = ajuste_lineal(logM, logD, 0, dlogD)
    err_pendiente = np.sqrt(cov[0, 0])
    valores_b.append(pendiente)
    errores_b.append(err_pendiente)

    A_fit = np.exp(ordenada)
    masas_np = np.array(masas)
    diametros_np = np.array(diametros)
    errores_d_np = np.array(errores_diametro)

    # Gráfico
    plt.errorbar(masas_np, diametros_np, yerr=errores_d_np,
                 fmt=marcadores[nombre_grupo], label=f"{nombre_grupo} (a={A_fit:.2f})",
                 color=colores[nombre_grupo], markersize=10, capsize=5)

    rango_masa = np.linspace(min(masas_np), max(masas_np), 100)
    pred = modelo_potencia(rango_masa, A_fit, pendiente)
    plt.plot(rango_masa, pred, color=colores[nombre_grupo], linewidth=2)

plt.xlabel("Masa (g)")
plt.ylabel("Diámetro promedio (cm)")
plt.grid()
plt.legend()
plt.savefig("diametro_vs_masa.png", dpi=300)
plt.close()

# --- Gráfico log-log global ---

plt.figure(figsize=(10, 7))
for nombre_grupo, muestras in grupos.items():
    logM, logD, dlogD = preparar_loglog(muestras)
    plt.errorbar(logM, logD, yerr=dlogD, fmt=marcadores[nombre_grupo],
                 color=colores[nombre_grupo], markersize=10, capsize=5, label=nombre_grupo)

logM_all = np.concatenate([preparar_loglog(m)[0] for m in grupos.values()])
logD_all = np.concatenate([preparar_loglog(m)[1] for m in grupos.values()])
dlogD_all = np.concatenate([preparar_loglog(m)[2] for m in grupos.values()])

pend, inter, _ = ajuste_lineal(logM_all, logD_all, 0, dlogD_all)
x_fit = np.linspace(min(logM_all), max(logM_all), 100)
plt.plot(x_fit, pend * x_fit + inter, 'k--', label="Ajuste global")

plt.xlabel("log(Masa)")
plt.ylabel("log(Diámetro)")
plt.legend()
plt.grid(True, which="both", linestyle="--")
plt.savefig("loglog_diametro_vs_masa.png", dpi=300)
plt.close()

# --- Gráfico b vs gramaje ---

plt.figure(figsize=(8, 6))
for nombre, g, dg, b, db in zip(grupos, gramajes_prom, errores_gramaje, valores_b, errores_b):
    plt.errorbar(g, b, xerr=dg, yerr=db, fmt=marcadores[nombre], color=colores[nombre],
                 label=nombre, capsize=5, markersize=10)

g_np = np.array(gramajes_prom)
b_np = np.array(valores_b)
dg_np = np.array(errores_gramaje)
db_np = np.array(errores_b)

pend, inter, cov = ajuste_lineal(g_np, b_np, dg_np, db_np)
err_pend, err_inter = np.sqrt(np.diag(cov))
x_vals = np.linspace(min(g_np), max(g_np), 100)
y_fit, _ = lineal_ajustada(x_vals, pend, inter, 0, cov)
y_sup, _ = lineal_ajustada(x_vals, pend + err_pend, inter + err_inter, 0, cov)
y_inf, _ = lineal_ajustada(x_vals, pend - err_pend, inter - err_inter, 0, cov)

plt.plot(x_vals, y_fit, 'k--', label="Tendencia lineal")
plt.fill_between(x_vals, y_inf, y_sup, color='gray', alpha=0.3)
plt.xlabel("Gramaje promedio (g/m²)")
plt.ylabel("Exponente b")
plt.grid()
plt.legend()
plt.savefig("b_vs_gramaje.png", dpi=300)
plt.close()

# --- Exportar CSV ---

df = pd.DataFrame({
    "Grupo": list(grupos.keys()),
    "Gramaje promedio (g/m²)": gramajes_prom,
    "ΔGramaje": errores_gramaje,
    "b": valores_b,
    "Δb": errores_b
})
df.to_csv("resumen_exponente_vs_gramaje.csv", index=False)
print("Archivo CSV exportado: resumen_exponente_vs_gramaje.csv")

# --- Resultados adicionales ---

print("\n--- Gramaje estimado por ajuste M = G·A (con ODR) ---")
for nombre_grupo, muestras in grupos.items():
    G_cm2, dG_cm2 = gramaje_por_ajuste_area_masa(muestras)
    print(f"{nombre_grupo}: G = {G_cm2*10000:.1f} ± {dG_cm2*10000:.1f} g/m²")

print("\n--- Ajuste no lineal con curve_fit (D = A·M^b) ---")
for nombre_grupo, muestras in grupos.items():
    A, dA, b, db = ajustar_con_curve_fit(muestras)
    print(f"{nombre_grupo}: A = {A:.2f} ± {dA:.2f}, b = {b:.3f} ± {db:.3f}")
