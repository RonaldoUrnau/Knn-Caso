# Caso 2 — Nodo Internet: Predicción de Bajas con KNN

Caso práctico de **Minería de Datos / Machine Learning aplicado a Negocios**. 

**Nodo Internet** es un proveedor de internet residencial (ISP) que enfrenta una pérdida de ~2.300 clientes por mes (tasa de *churn* del 8% a 90 días). Para contener las bajas, la empresa dispone de un presupuesto operativo limitado que solo permite realizar **420 llamadas telefónicas** de retención ofreciendo un descuento agresivo.

El objetivo del proyecto es estimar con precisión la probabilidad de que un cliente se dé de baja en los próximos 90 días para priorizar el cupo de llamadas y maximizar el retorno económico. Para ello, se implementó un modelo de **KNN (K-Nearest Neighbors) programado desde cero en NumPy** y se contrastó contra **Naive Bayes** de scikit-learn.

---

## 📁 Estructura del Repositorio

```text
.
├── caso2_nodo.ipynb                  # Notebook guía interactivo con los 24 pasos resueltos
├── datos/
│   ├── clientes_historico.csv        # 20.000 clientes con baja_90d conocida (entrenamiento y test)
│   └── clientes_campania.csv         # 6.000 clientes activos de octubre a evaluar
├── campania_octubre.csv              # Entregable final: id_cliente, prob_baja, contactar (exactamente 420)
├── src/
│   ├── preparacion.py                # Limpieza, normalización de textos y feature engineering
│   ├── knn.py                        # EscaladorEstandar y KNNClasificador desarrollados desde cero en NumPy
│   └── negocio.py                    # Economía de la campaña, optimización de umbral y curvas de ganancia
├── tests/                            # Batería de pruebas unitarias automatizadas (36 tests)
│   ├── test_preparacion.py
│   ├── test_knn.py
│   └── test_negocio.py
├── conftest.py                       # Configuración de imports de pytest
├── pytest.ini                        # Configuración de descubrimiento de pruebas
├── requirements.txt                  # Dependencias del proyecto
├── INFORME_EJECUTIVO_MARTIN_AQUINO.md # Reporte ejecutivo con recomendaciones estratégicas
└── informe_ejecutivo.pdf             # Informe formal en PDF para el comité de dirección
```

---

## 💼 La Consigna de Negocio y Reglas Clave

El gerente comercial (**Martín Aquino**) propone una regla intuitiva: *"Llamar a los clientes con contrato mensual y 2 o más reclamos técnicos"*. El equipo de ciencia de datos debe evaluar financieramente esta heurística y reemplazarla por un modelo analítico que determine a quién contactar.

### 1. Trampa de Fuga de Datos (*Data Leakage*)
La columna `motivo_baja` solo existe para los clientes que ya cancelaron el servicio. Incluirla en el entrenamiento generaría un modelo ficticio e inutilizable en producción, ya que al evaluar clientes vigentes dicha columna no existe. Se excluyó estrictamente del modelado.

### 2. Economía y Probabilidad de Equilibrio ($p^*$)
Los parámetros acordados con Finanzas y Fidelización son:
* **Costo por llamada telefónica:** Gs. 10.000
* **Costo de la promoción ofrecida:** Gs. 200.000 (aceptada por el 40% si el cliente pensaba irse, o 60% si pensaba quedarse)
* **Margen salvado por cliente retenido:** Gs. 1.000.000

A partir de estos valores se derivan los resultados esperados:
* **Resultado si el cliente se iba ($G^+$):**  
  $$G^+ = 0.40 \times (1.000.000 - 200.000) - 10.000 = \mathbf{+Gs.\ 310.000}$$
* **Resultado si el cliente se quedaba ($G^-$):**  
  $$G^- = -(0.60 \times 200.000) - 10.000 = \mathbf{-Gs.\ 130.000}$$
* **Probabilidad de equilibrio ($p^*$):**  
  $$p^* = \frac{-G^-}{G^+ - G^-} = \frac{130.000}{310.000 - (-130.000)} \approx \mathbf{29.55\%}$$

> **Criterio de decisión:** Contactar a un cliente solo es rentable si su riesgo predicho de fuga supera $\approx 30\%$.

### 3. ¿Por qué KNN como Algoritmo Principal?
Además de un sólido rendimiento probabilístico, KNN ofrece **interpretabilidad comercial directa**: permite generar fichas de fidelización para los operadores del call center mostrando los clientes históricos más parecidos (vecinos cercanos) y qué motivos declararon al darse de baja, facilitando una propuesta personalizada (precio, visita técnica o mejora de velocidad).

---

## 📈 Principales Resultados

| Métrica / Dimensión | Regla Manual Comercial | Modelo KNN Óptimo ($k=81$, Manhattan, Distance) |
| :--- | :---: | :---: |
| **Puntaje de Brier (Calibración)** | — | **0.0623** *(vs. 0.1077 en Naive Bayes)* |
| **Contactados Campaña Octubre** | Criterio rígido | **420 clientes residenciales** (tope operativo) |
| **Precisión Estimada (% en fuga real)** | Baja | **51.25%** (~215 clientes en riesgo real contactados) |
| **Resultado Financiero Neto** | Pérdidas económicas | **+Gs. 40.101.779** de ganancia neta esperada |

*Nota sobre Naive Bayes:* Al violar el supuesto de independencia entre variables correlacionadas (como velocidad, reclamos y horas sin servicio), Naive Bayes polariza las probabilidades en los extremos (0 y 1), generando un Brier Score deficiente y requiriendo un umbral extremo (>0.99) para no inundar el call center de falsos positivos. KNN produce probabilidades suaves y bien calibradas.

---

## 🚀 Cómo Reproducir el Proyecto

### 1. Clonar el repositorio y configurar el entorno
```bash
# Crear entorno virtual
python -m venv .venv

# Activar entorno virtual
# En Windows:
.venv\Scripts\activate
# En Linux/macOS:
source .venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

### 2. Ejecutar las pruebas unitarias
Para validar que todos los módulos de `src/` (`preparacion.py`, `knn.py` y `negocio.py`) cumplen con las especificaciones:
```bash
python -m pytest -q
```
*(36 pruebas unitarias implementadas y aprobadas al 100%).*

### 3. Explorar el Notebook
Abrir Jupyter Lab o Notebook en la raíz del repositorio:
```bash
jupyter lab caso2_nodo.ipynb
```
El notebook contiene la resolución de los 24 pasos del caso, gráficos de calibración, curva de ganancia vs. umbral, importancia de variables por permutación y la exportación de [campania_octubre.csv](campania_octubre.csv).

---

## 📋 Requisitos

* Python 3.10+
* `numpy >= 1.26`
* `pandas >= 2.1`
* `scikit-learn >= 1.4`
* `scipy >= 1.11`
* `matplotlib >= 3.8`
* `pytest >= 8.0`
* `jupyterlab >= 4.0`
