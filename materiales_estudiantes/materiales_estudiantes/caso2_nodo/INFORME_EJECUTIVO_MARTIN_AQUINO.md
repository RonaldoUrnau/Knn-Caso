# 📄 INFORME EJECUTIVO · NODO INTERNET

**Para:** Martín Aquino (Dirección Comercial y General)  
**De:** Equipo de Ciencia de Datos & Fidelización  
**Fecha:** Octubre 2026  
**Alcance:** Cartera Residencial de Clientes Activos (Base de 6.000 clientes para campaña de retención)  
**Entregables adjuntos:** `informe_ejecutivo.pdf`, `campania_octubre.csv`, `caso2_nodo.ipynb`

---

## 1. Resumen Ejecutivo y Diagnóstico del Negocio

Nodo Internet enfrenta una pérdida mensual recurrente de aproximadamente **2.300 clientes residenciales**, lo que equivale a una **tasa de baja (churn) del 7.68% en 90 días**. Para contener esta fuga, el equipo de Fidelización planifica una campaña de contacto proactivo con un cupo operativo máximo de **420 llamadas telefónicas** para ofrecer planes de retención y descuentos.

* **El peligro de la regla manual comercial:** La regla empírica sugerida por el área de ventas (*"llamar a clientes con contrato mensual y $\ge 2$ reclamos técnicos"*) demostró ser **económicamente deficitaria**: solo alcanza una precisión del **22.74%**, ubicándose por debajo del punto de equilibrio económico (**29.55%**). Aplicar esta regla hubiera generado **pérdidas netas por Gs. -10.440.000** en el conjunto de prueba y más de Gs. -28.000.000 en entrenamiento, debido a que regala descuentos costosos (Gs. 200.000) a clientes que no tenían intenciones reales de marcharse.
* **El éxito del modelo predictivo (KNN):** El algoritmo de Machine Learning optimizado alcanza una precisión del **51.25%** en los clientes contactados (un **lift de más de 6.6 veces** respecto a la tasa natural de la base), duplicando con creces el umbral de rentabilidad.
* **Impacto Económico Final:** La campaña de octubre basada en nuestro modelo proyecta una **ganancia neta de Gs. 40.101.779**, salvando a más de 215 clientes residenciales de alto valor y garantizando el uso eficiente de los recursos del call center.

---

## 2. Comparativa Económica de Estrategias

Evaluación rigurosa fuera de muestra (conjunto de Test de 4.847 clientes):

| Estrategia Evaluada | Contactados | % de Cartera | Precisión (% Fuga Real) | Ganancia Neta Total (Gs.) | Ganancia / Cliente | Veredicto Comercial |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. No hacer nada (Línea Base)** | 0 | 0.0% | 0.0% | Gs. 0 | Gs. 0 | Pérdida pasiva de clientes |
| **2. Regla Manual del Gerente** | 324 | 6.68% | 22.74% | **Gs. -10.440.000** | -Gs. 2.154 | **DEFICITARIA (PÉRDIDA)** |
| **3. Modelo Naive Bayes** | 257 | 5.30% | 43.97% | Gs. 20.270.000 | +Gs. 4.182 | Rentable pero subóptimo |
| **4. Modelo KNN (Seleccionado)** | **281** | **5.80%** | **51.25%** | **Gs. 26.830.000** | **+Gs. 5.535** | **MÁXIMA RENTABILIDAD** |

*Proyección a la campaña de 6.000 clientes con el tope de 420 contactos:*
* **Clientes salvados proyectados ($G^+$):** 215.2 clientes (margen retenido neto: Gs. 66.720.000).
* **Costos por contactos no efectivos / descuentos ($G^-$):** 204.8 clientes (costo neto: Gs. -26.618.000).
* **GANANCIA NETA TOTAL ESPERADA:** **Gs. 40.101.779**.

---

## 3. Los 4 Factores Críticos que Disparan la Cancelación

El análisis de datos y la importancia por permutación revelaron los cuatro motivos principales de fuga:

1. **Consulta de Baja en la Web (Alerta Digital Temprana):**
   * Los clientes que visitaron la sección de cancelación en la web en los últimos 30 días tienen una tasa de baja del **38.85%** (frente a 5.96% en quienes no entraron). Es la señal individual más potente de intención de retiro.
2. **Presión Competitiva en Zona Este (Sensibilidad al Precio):**
   * En la Zona Este, los clientes que abonan más de Gs. 280.000 mensuales se dan de baja en un **28.22%** (más de 4 veces el promedio general). El análisis cualitativo confirma que competidores locales están captando clientes mediante planes agresivos de fibra a menores precios.
3. **Vencimiento de Permanencia Contractual:**
   * Al expirar el contrato (o faltar $\le 3$ meses para el vencimiento), la tasa de abandono se duplica (del 3.80% con permanencia lejana al **10.05%** sin penalidad). El cliente queda libre de cambiar de proveedor.
4. **Acumulación de Problemas Técnicos:**
   * Cada reclamo técnico aumenta exponencialmente la insatisfacción: con 0 reclamos el churn es 5.7%; con 2 reclamos sube al 13.8%; con 3 al 17.9%; y con 4 o más supera el **25.6%**.

---

## 4. Protocolo Operativo para Fidelización (Ofertas Personalizadas)

Gracias a la capacidad explicativa del algoritmo KNN, los operadores del call center no llamarán a ciegas ofreciendo el mismo descuento genérico, sino que contarán con una **Ficha de Diagnóstico Causal** basada en el comportamiento de los clientes del pasado más parecidos:

* **Caso A · Riesgo Técnico (Vecinos se fueron por fallas o cliente con $\ge 2$ reclamos):**
  * *Acción:* **Oferta Técnica Prioritaria**. Agendamiento de visita técnica en 24 horas sin costo para reemplazo de módem o revisión de acometida, más bonificación del 20% en la factura por los inconvenientes.
* **Caso B · Riesgo Competitivo (Zona Este o planes $\ge$ Gs. 280.000):**
  * *Acción:* **Oferta Comercial Agresiva**. Descuento promocional de Gs. 200.000 o duplicación gratuita de velocidad durante 6 meses para blindar al cliente frente a tarifas de la competencia.
* **Caso C · Riesgo por Fin de Contrato:**
  * *Acción:* **Oferta de Vinculación**. Descuento fidelización escalonado en 3 facturas a cambio de renovación de contrato a 12 meses.

---

## 5. Recomendaciones Estratégicas y Próximos Pasos

1. **Ejecución Inmediata de la Campaña:** Utilizar el archivo entregable [`campania_octubre.csv`](file:///c:/tareita_roni_2/Knn_Caso/materiales_estudiantes/materiales_estudiantes/caso2_nodo/campania_octubre.csv), que contiene a los 420 clientes residenciales priorizados por riesgo económico, sin inclusión de empleados ni registros duplicados.
2. **Auditoría Científica Mediante Grupo de Control (Test A/B):** De los 420 clientes seleccionados, apartar aleatoriamente un **10% (42 clientes) como grupo de control** a los cuales no se llamará. Al término del trimestre, comparar la tasa de baja entre el grupo contactado y el grupo de control para aislar el *efecto causal puro* y validar si se cumplió la tasa de efectividad del 40%.
3. **Monitoreo Continuo de Deriva (Concept Drift):** Implementar un monitoreo mensual del Índice de Estabilidad Poblacional (PSI) y de los motivos de baja para recalibrar el modelo ante cambios macroeconómicos o nuevas ofensivas tarifarias de la competencia.
