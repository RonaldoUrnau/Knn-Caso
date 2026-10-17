"""Preparación de los datos de clientes de Nodo Internet.

Completá las dos funciones con pandas y la biblioteca estándar.
Verificá tu trabajo con:  python -m pytest -q tests/test_preparacion.py
"""
import unicodedata

import numpy as np
import pandas as pd

METODOS_VALIDOS = ["debito_automatico", "tarjeta_credito", "transferencia", "efectivo"]
DIAS_POR_MES = 30.44


def normalizar_metodo_pago(valor):
    """Lleva las variantes con que el CRM guarda el medio de pago a uno de METODOS_VALIDOS.

    Ejemplos: "Débito automático", "DEBITO AUTOMATICO" y "Debito_Automatico" -> "debito_automatico";
    "Tarjeta de crédito" y "tarjeta credito" -> "tarjeta_credito"; "transferencia bancaria" -> "transferencia";
    "efectivo " -> "efectivo".
    Si el valor no corresponde a ninguno de los cuatro, lanzar ValueError.
    """
    if not isinstance(valor, str):
        raise ValueError("El valor debe ser una cadena de texto")
    
    v = unicodedata.normalize("NFKD", valor).encode("ascii", "ignore").decode("utf-8")
    v = v.lower().strip()
    
    if "debito" in v:
        return "debito_automatico"
    elif "tarjeta" in v:
        return "tarjeta_credito"
    elif "transferencia" in v:
        return "transferencia"
    elif "efectivo" in v:
        return "efectivo"
    else:
        raise ValueError(f"Método de pago no reconocido: {valor}")


def agregar_variables(df):
    """Devuelve una COPIA de df (sin modificar el original) con metodo_pago normalizado y tres columnas nuevas.

    Todas se calculan a la fecha_corte de cada fila, nunca a la fecha de hoy:
    - antiguedad_meses: días entre fecha_alta y fecha_corte, divididos por DIAS_POR_MES.
    - meses_hasta_fin_permanencia: días entre fecha_corte y fin_permanencia, divididos por DIAS_POR_MES.
      Vale 0 en los contratos mensuales (no tienen fecha de fin de permanencia).
    - sin_permanencia_3m: 1 si el cliente puede irse sin penalidad en los próximos 3 meses, es decir,
      si su contrato es mensual o si meses_hasta_fin_permanencia <= 3; 0 en otro caso (entero).
    """
    df_nuevo = df.copy()
    
    df_nuevo["metodo_pago"] = df_nuevo["metodo_pago"].apply(normalizar_metodo_pago)
    
    fecha_corte = pd.to_datetime(df_nuevo["fecha_corte"])
    fecha_alta = pd.to_datetime(df_nuevo["fecha_alta"])
    df_nuevo["antiguedad_meses"] = (fecha_corte - fecha_alta).dt.days / DIAS_POR_MES
    
    fin_permanencia = pd.to_datetime(df_nuevo["fin_permanencia"])
    dias_permanencia = (fin_permanencia - fecha_corte).dt.days.fillna(0)
    dias_permanencia = np.where(df_nuevo["tipo_contrato"] == "mensual", 0.0, np.maximum(0.0, dias_permanencia))
    df_nuevo["meses_hasta_fin_permanencia"] = dias_permanencia / DIAS_POR_MES
    
    condicion_sin_perm = (df_nuevo["tipo_contrato"] == "mensual") | (df_nuevo["meses_hasta_fin_permanencia"] <= 3.0)
    df_nuevo["sin_permanencia_3m"] = condicion_sin_perm.astype(int)
    
    return df_nuevo
