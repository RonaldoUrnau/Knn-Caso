"""Evaluación económica de la campaña de retención de Nodo Internet.

Completá las funciones. Solo NumPy. Verificá tu trabajo con:  python -m pytest -q tests/test_negocio.py
"""
import numpy as np

# Supuestos acordados por Finanzas y Fidelización (ver el enunciado del caso)
SUPUESTOS = {
    "costo_contacto": 10_000,            # Gs. por cliente contactado
    "costo_descuento": 200_000,          # Gs. por cliente que acepta el descuento
    "valor_cliente_retenido": 1_000_000,  # Gs. de margen de un cliente que se iba a ir y se queda
    "acepta_si_se_iba": 0.40,            # proporción que acepta la oferta entre los que se iban a ir
    "acepta_si_se_quedaba": 0.60,        # proporción que acepta la oferta entre los que se iban a quedar
}


def resultado_por_contacto(s=SUPUESTOS):
    """Devuelve una tupla (si_se_iba, si_se_quedaba) con el resultado esperado, en guaraníes, de contactar a:
    - un cliente que se iba a ir (si acepta, se queda y deja su margen, menos el descuento), y
    - un cliente que se iba a quedar (si acepta, la empresa resigna el descuento).
    En los dos casos se paga el costo del contacto. No contactar tiene resultado 0.
    """
    si_se_iba = s["acepta_si_se_iba"] * (s["valor_cliente_retenido"] - s["costo_descuento"]) - s["costo_contacto"]
    si_se_quedaba = - (s["acepta_si_se_quedaba"] * s["costo_descuento"]) - s["costo_contacto"]
    return float(si_se_iba), float(si_se_quedaba)


def probabilidad_de_equilibrio(s=SUPUESTOS):
    """Probabilidad de baja p a partir de la cual contactar tiene resultado esperado positivo:
    el valor de p que hace p · si_se_iba + (1 - p) · si_se_quedaba = 0.
    """
    si_se_iba, si_se_quedaba = resultado_por_contacto(s)
    p = -si_se_quedaba / (si_se_iba - si_se_quedaba)
    return float(p)


def ganancia_campania(y_real, contactar, s=SUPUESTOS):
    """Ganancia total, en guaraníes, de contactar a los clientes marcados con True en `contactar`,
    sabiendo quiénes se fueron realmente (`y_real`, 1 = se fue). Devuelve un float.
    """
    y_arr = np.asarray(y_real)
    contactar_arr = np.asarray(contactar, dtype=bool)
    
    si_se_iba, si_se_quedaba = resultado_por_contacto(s)
    
    n_se_iban = np.sum((y_arr == 1) & contactar_arr)
    n_se_quedaban = np.sum((y_arr == 0) & contactar_arr)
    
    ganancia = n_se_iban * si_se_iba + n_se_quedaban * si_se_quedaba
    return float(ganancia)


def mejor_umbral(y_real, prob, s=SUPUESTOS, max_contactos=None):
    """Busca el umbral u que maximiza la ganancia al contactar a todos los clientes con prob >= u.

    - Los umbrales candidatos son los valores distintos de `prob`.
    - Si max_contactos no es None, se descartan los umbrales que contactan a más clientes.
    - Devuelve un dict {"umbral": u, "ganancia": g, "contactados": n}.
    - Si ningún umbral da ganancia positiva, devuelve {"umbral": np.inf, "ganancia": 0.0, "contactados": 0}.
    """
    y_arr = np.asarray(y_real)
    prob_arr = np.asarray(prob)
    umbrales = np.unique(prob_arr)
    
    mejor_g = 0.0
    mejor_u = np.inf
    mejor_n = 0
    
    for u in np.sort(umbrales)[::-1]:
        contactar = prob_arr >= u
        n = int(np.sum(contactar))
        if max_contactos is not None and n > max_contactos:
            continue
        g = ganancia_campania(y_arr, contactar, s)
        if g > mejor_g:
            mejor_g = g
            mejor_u = float(u)
            mejor_n = n
            
    return {"umbral": mejor_u, "ganancia": float(mejor_g), "contactados": int(mejor_n)}


def lift(y_real, prob, fraccion):
    """Tasa de positivos entre la `fraccion` de clientes con mayor prob, dividida por la tasa general.

    Ejemplo: lift(y, prob, 0.07) compara la tasa de baja del 7% de mayor riesgo con la tasa de toda la base.
    La cantidad de clientes del grupo es round(fraccion · n), con un mínimo de 1.
    """
    y_arr = np.asarray(y_real)
    prob_arr = np.asarray(prob)
    n = len(y_arr)
    k = max(1, int(round(fraccion * n)))
    
    orden = np.argsort(-prob_arr)
    top_k_y = y_arr[orden[:k]]
    
    tasa_grupo = np.mean(top_k_y)
    tasa_general = np.mean(y_arr)
    
    if tasa_general == 0:
        return 0.0
    return float(tasa_grupo / tasa_general)
