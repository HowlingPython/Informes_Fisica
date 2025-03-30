\documentclass[12pt,a4]{article}
\usepackage{booktabs}
\usepackage[left=1.8cm,right=1.8cm,top=32mm,columnsep=20pt]{geometry}
\usepackage{fontspec}
\usepackage[spanish, es-tabla, es-nodecimaldot]{babel}
\usepackage{amsmath, amsfonts}
\usepackage{float}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{multicol}
\usepackage{tocloft}
\usepackage[sorting=none]{biblatex}
\addbibresource{main.bib}

\title{Leyes de escala en bollos de papel}
\author{Federico Tomás Moroni, Nicolás Ezequiel López, Gaia Tornelli\\[2mm]
\small Universidad de San Andrés}
\date{1er Semestre 2025}

\begin{document}
\maketitle
\begin{abstract}
En esta práctica se investigó la relación entre el diámetro de los bollos de papel y la masa de la hoja utilizada, explorando una posible ley de escala del tipo \( D = A \cdot M^b \). Se midieron áreas, masas, volúmenes y diámetros para tres tipos de papel con diferentes gramajes, propagando las incertidumbres correspondientes. Se obtuvieron exponentes característicos mediante ajustes logarítmico-logarítmico (log-log) y no lineal. Asimismo, se estimó el gramaje del papel a partir de un ajuste ortogonal. Los resultados muestran que el valor del exponente \( b \) aumenta con el gramaje, lo que sugiere una influencia estructural del material en el proceso de compactación.
\end{abstract}

\newpage
\renewcommand{\cfttoctitlefont}{\Large\bfseries}
\renewcommand{\contentsname}{Contenido}
\setlength{\cftbeforesecskip}{4pt}
\setlength{\cftbeforesubsecskip}{2pt}
\tableofcontents
\newpage

\section{Introducción}

Las leyes de escala describen cómo una variable física depende de otra mediante una relación de tipo potencia: \( y = A x^b \). Este tipo de vínculo está presente en múltiples contextos físicos, como los fractales, el crecimiento biológico, los materiales comprimidos y las redes complejas \cite{clauset}.

En particular, el estudio de materiales comprimibles, como el papel arrugado, permite explorar la relación entre propiedades geométricas (como el diámetro de un bollo) y propiedades físicas (como la masa). Estudios similares en polímeros, espumas y estructuras plegadas han demostrado que el exponente de la ley de escala puede depender significativamente de la rigidez y la estructura del material.

Asimismo, esta temática se relaciona con aplicaciones prácticas como el diseño de empaques, el reciclaje de materiales y la ingeniería de estructuras colapsables. La identificación de patrones empíricos en sistemas cotidianos facilita la conexión entre conceptos estadísticos y experiencias tangibles, promoviendo una comprensión más profunda de fenómenos aparentemente simples.

El objetivo principal de este trabajo es verificar una ley de escala del tipo \( D = A \cdot M^b \) entre el diámetro \( D \) del bollo y la masa \( M \) del papel, y analizar cómo varía el exponente \( b \) con el gramaje del material. Además, se reconoce que el análisis de este tipo de relaciones empíricas debe realizarse con cautela. Como advierten Clauset, Shalizi y Newman (2009), en muchos casos se asume erróneamente que la linealidad en escala logarítmica confirma una ley de potencia, cuando en realidad dicho comportamiento puede corresponder a otras distribuciones con colas pesadas. Por ello, este trabajo adopta un enfoque experimental y comparativo, priorizando la coherencia física y la validación empírica del ajuste, más allá de una interpretación estadística estricta de las distribuciones.



\section{Desarrollo experimental}

Se utilizaron tres tipos de papel: liviano, medio y pesado. Para cada uno se cortaron hojas de distintos tamaños y se formaron bollos de manera manual, procurando mantener constante la técnica de compactación.

Se realizaron las siguientes mediciones:

\begin{itemize}
    \item Longitud \( L \) y ancho \( W \) de la hoja, con una regla milimetrada (\( \delta = \SI{0.05}{cm} \)).
    \item Masa \( M \), con una balanza digital de precisión (\( \delta = \SI{0.01}{g} \)).
    \item Diámetro \( D \) del bollo, con una regla, tomando cinco repeticiones por muestra.
\end{itemize}

El área del papel se calculó mediante:

\[
A = L \cdot W
\]

Y el volumen del bollo, asumiendo una forma esférica ideal, se estimó como:

\[
V = \frac{4}{3} \pi \left( \frac{D}{2} \right)^3
\]

Esta aproximación supone una esfericidad razonable. Si bien no representa fielmente la geometría real del bollo, resulta adecuada para una primera estimación del volumen ocupado.

El gramaje del papel se estimó mediante la expresión:

\[
G = \frac{10000 \cdot M}{A}
\]

Esta magnitud representa la densidad superficial del papel (en g/m\(^2\)) y permite clasificarlo de acuerdo con estándares comerciales.

Se propagaron los errores en todas las magnitudes calculadas. En particular, para el volumen:

\[
\delta V = \frac{\pi D^2}{2} \cdot \delta D
\]

Para reducir el sesgo derivado de la dispersión aleatoria, se repitió la medición del diámetro al menos seis veces por muestra, calculando el promedio y su error estándar. Además, se aplicó un ajuste ortogonal (ODR, por sus siglas en inglés) al modelo \( M = G \cdot A \), con el fin de obtener una estimación más precisa del gramaje.

\section{Resultados y análisis}

\subsection{Relación entre diámetro y masa}

Para cada tipo de papel se graficó el diámetro promedio de los bollos en función de su masa (véase la Figura~\ref{fig:diametro_vs_masa}). Aunque la relación no es lineal en escala aritmética, se observa una clara tendencia creciente.

\begin{figure}[H]
    \centering
    \includegraphics[width=0.45\textwidth]{diametro_vs_masa.png}
    \caption{Diámetro promedio de los bollos en función de la masa, diferenciados por grupo.}
    \label{fig:diametro_vs_masa}
\end{figure}

Por ejemplo, una muestra del grupo medio, con un área \( A = \SI{300}{cm^2} \) y una masa \( M = \SI{4.9}{g} \), produjo un bollo de \( D = \SI{4.2}{cm} \), en concordancia con el modelo teórico propuesto.

La transformación logarítmica permite linealizar la relación (véase la Figura~\ref{fig:loglog}) y aplicar un ajuste ponderado para estimar los parámetros \( b \) y \( A \):

\begin{figure}[H]
    \centering
    \includegraphics[width=0.45\textwidth]{loglog_diametro_vs_masa.png}
    \caption{Gráfico log-log del diámetro en función de la masa. La linealidad sugiere una ley de escala.}
    \label{fig:loglog}
\end{figure}

\subsection{Comparación de métodos de ajuste}

Ambos métodos —el ajuste lineal sobre coordenadas logarítmicas (log-log) y el ajuste no lineal directo— arrojaron resultados consistentes, lo que refuerza la validez del modelo propuesto.

\begin{table}[H]
\centering
\begin{tabular}{@{}lccc@{}}
\toprule
\textbf{Grupo} & \textbf{\( b \)} (log-log) & \textbf{\( b \)} (no lineal) & \textbf{\( A \)} \\
\midrule
Liviana & 0{,}481 ± 0{,}028 & 0{,}447 ± 0{,}014 & 1{,}96 ± 0{,}05 \\
Media   & 0{,}368 ± 0{,}025 & 0{,}368 ± 0{,}025 & 2{,}01 ± 0{,}04 \\
Pesada  & 0{,}388 ± 0{,}012 & 0{,}394 ± 0{,}010 & 1{,}94 ± 0{,}04 \\
\bottomrule
\end{tabular}
\caption{Parámetros obtenidos mediante ambos métodos de ajuste, según el tipo de papel.}
\end{table}

El bajo nivel de incertidumbre en los parámetros ajustados respalda la calidad del modelo, en especial en el grupo de papel pesado.

\textit{Consideración metodológica.} Como advierten Clauset, Shalizi y Newman (2009), la simple linealidad observada en un gráfico log-log no constituye una prueba concluyente de que los datos siguen una ley de potencia. Este comportamiento puede ser compatible con otras distribuciones de cola pesada, como la log-normal o la exponencial truncada.

En este trabajo, la representación logarítmica se utiliza como herramienta para facilitar el ajuste de un modelo empírico. No se pretende validar estadísticamente la forma funcional de la ley ni excluir modelos alternativos. La justificación de la relación \( D = A \cdot M^b \) se fundamenta en su adecuación experimental y en la coherencia física observada en los valores del exponente \( b \).

\subsection{Interpretación del exponente \( b \)}

El exponente \( b \) indica cómo escala el diámetro del bollo con respecto a la masa del papel. Un valor de \( b = \frac{1}{3} \) correspondería a un comportamiento de densidad constante, es decir, a un escalado volumétrico puro. Sin embargo, los valores obtenidos son superiores a \( \frac{1}{3} \), lo cual sugiere que la densidad del bollo disminuye a medida que aumenta la masa, lo que indica un proceso de compactación no proporcional.

\begin{figure}[H]
    \centering
    \includegraphics[width=0.45\textwidth]{b_vs_gramaje.png}
    \caption{Variación del exponente \( b \) en función del gramaje del papel. Se observa una tendencia creciente.}
    \label{fig:b_vs_gramaje}
\end{figure}

Este comportamiento sugiere que, a mayor rigidez del material (mayor gramaje), la resistencia del papel al plegado afecta negativamente la eficiencia del proceso de compactación.

\subsection{Estimación del gramaje por ajuste ortogonal}

El modelo \( M = G \cdot A \) permitió estimar el gramaje del papel de forma más precisa y robusta, mediante un ajuste ortogonal por mínimos cuadrados (ODR).

\begin{table}[H]
\centering
\begin{tabular}{@{}lcc@{}}
\toprule
\textbf{Grupo} & \( G \) (g/m²) & \( \Delta G \) \\
\midrule
Liviana & 83{,}9 & 0{,}6 \\
Media   & 165{,}5 & 0{,}5 \\
Pesada  & 250{,}7 & 4{,}3 \\
\bottomrule
\end{tabular}
\caption{Estimación del gramaje mediante ajuste ODR para cada tipo de papel.}
\end{table}

Los valores obtenidos son coherentes con los esperados para cada tipo de papel y presentan una dispersión reducida en comparación con las estimaciones obtenidas mediante promedios simples.

\subsection{Limitaciones del modelo}

\begin{itemize}
    \item Se asumió una forma esférica ideal para los bollos, lo cual puede introducir errores sistemáticos al estimar el volumen real, ya que la geometría verdadera es irregular.
    
    \item La fuerza aplicada durante la compactación del papel no fue cuantificada ni controlada, lo que introduce una fuente significativa de variabilidad entre muestras.
    
    \item El rango de masas considerado fue relativamente acotado, por lo que los resultados podrían no extrapolarse adecuadamente a hojas mucho más grandes o más pequeñas.
    
    \item La estimación del gramaje se basa en la suposición de homogeneidad del papel, la cual no fue verificada por métodos microscópicos ni estructurales.
\end{itemize}


\subsection{Síntesis de observaciones}

\begin{itemize}
    \item La ley de escala \( D = A \cdot M^b \) representa adecuadamente el comportamiento observado en los datos experimentales.
    
    \item El valor del exponente \( b \) tiende a aumentar con el gramaje del papel, lo que sugiere que materiales más rígidos presentan una compactación menos eficiente.
    
    \item El método de ajuste no lineal se considera más apropiado, ya que permite incorporar directamente las incertidumbres en ambas variables.
    
    \item La estimación del gramaje mediante ajuste ortogonal mostró menor dispersión y mayor robustez que la obtenida por promedio directo.
\end{itemize}

\section{Conclusión}

Los resultados obtenidos confirman que el diámetro de los bollos de papel escala con la masa mediante una ley de tipo potencia, con exponentes \( b \) comprendidos entre 0{,}36 y 0{,}48, en función del gramaje. Esto indica que el material no se compacta de forma isotrópica, y que la estructura del papel desempeña un papel fundamental en el proceso de arrugamiento.

Este trabajo permitió aplicar herramientas de análisis experimental, propagación de incertidumbres y comparación de modelos de ajuste. Asimismo, demuestra cómo conceptos estadísticos pueden integrarse al estudio de fenómenos cotidianos con rigurosidad física.

Cabe señalar que el presente estudio no tiene como objetivo validar rigurosamente una distribución de tipo ley de potencia en el sentido estadístico estricto. Siguiendo las advertencias de Clauset et al. (2009), se reconoce que la presencia de una recta en escala log-log no constituye evidencia concluyente de una ley de potencia. No obstante, en este contexto experimental, el modelo \( D = A \cdot M^b \) se considera adecuado para describir la relación entre la masa y el diámetro de los bollos, y proporciona una interpretación física coherente del exponente \( b \).


\appendix
\section{Fórmulas utilizadas y propagación de errores}

A continuación se detallan las expresiones empleadas para el cálculo de las magnitudes físicas y la propagación de sus respectivas incertidumbres.

\begin{itemize}
    \item \textbf{Área del papel y su incertidumbre:}
    \[
    A = L \cdot W, \quad
    \delta A = \sqrt{(W \cdot \delta L)^2 + (L \cdot \delta W)^2}
    \]

    \item \textbf{Volumen del bollo (supuesto esférico) y su incertidumbre:}
    \[
    V = \frac{4}{3} \pi \left( \frac{D}{2} \right)^3, \quad
    \delta V = \frac{\pi D^2}{2} \cdot \delta D
    \]

    \item \textbf{Gramaje del papel y su incertidumbre:}
    \[
    G = \frac{10000 \cdot M}{A}, \quad
    \delta G = G \cdot \sqrt{\left( \frac{\delta M}{M} \right)^2 + \left( \frac{\delta A}{A} \right)^2}
    \]
\end{itemize}

\printbibliography
\end{document}
