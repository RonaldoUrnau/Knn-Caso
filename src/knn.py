"""KNN y escalado estándar implementados desde cero.

Reglas:
- Los cálculos se hacen solo con NumPy. No se puede usar sklearn.neighbors, sklearn.preprocessing
  ni scipy.spatial.
- KNNClasificador hereda de ClassifierMixin y BaseEstimator solo para poder usarse dentro de un
  Pipeline y de GridSearchCV de scikit-learn.
- Nada de bucles sobre filas para calcular distancias: se calculan con operaciones vectorizadas,
  por lotes de `tamano_lote` filas para no llenar la memoria.

Verificá tu trabajo con:  python -m pytest -q tests/test_knn.py
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class EscaladorEstandar:
    """Estandariza cada columna: (x - media) / desvío.

    Atributos que crea fit: media_ y desvio_ (arrays de forma (n_columnas,)).
    - El desvío es el poblacional (np.std con ddof=0), igual que StandardScaler.
    - Si una columna es constante (desvío 0), se usa desvío 1 para no dividir por cero.
    """

    def fit(self, X, y=None):
        X_arr = np.asarray(X, dtype=float)
        self.media_ = np.mean(X_arr, axis=0)
        desvio = np.std(X_arr, axis=0, ddof=0)
        # Si una columna tiene desvío 0, usamos 1 para no dividir por cero
        desvio = np.where(desvio == 0.0, 1.0, desvio)
        self.desvio_ = desvio
        return self

    def transform(self, X):
        X_arr = np.asarray(X, dtype=float)
        return (X_arr - self.media_) / self.desvio_

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


class KNNClasificador(ClassifierMixin, BaseEstimator):
    """Clasificador de k vecinos más cercanos (búsqueda por fuerza bruta).

    Parámetros
    ----------
    k : int, por defecto 5
        Cantidad de vecinos.
    pesos : "uniforme" o "distancia"
        "uniforme": cada vecino vota con peso 1.
        "distancia": cada vecino vota con peso 1 / distancia. Si la consulta coincide exactamente con
        uno o más ejemplos (distancia 0), votan solo esos ejemplos, con peso 1 cada uno.
    metrica : "euclidea" o "manhattan"
    tamano_lote : int, por defecto 1000
        Cantidad de filas de consulta que se procesan juntas al calcular distancias.

    Atributos que crea fit
    ----------------------
    X_ : array (n_entrenamiento, n_variables) con los datos de entrenamiento, en float.
    classes_ : array (n_clases,) con las clases ordenadas (np.unique).
    y_codificada_ : array (n_entrenamiento,) con el índice de la clase de cada ejemplo en classes_.
    """

    def __init__(self, k=5, pesos="uniforme", metrica="euclidea", tamano_lote=1000):
        self.k = k
        self.pesos = pesos
        self.metrica = metrica
        self.tamano_lote = tamano_lote

    def fit(self, X, y):
        """KNN no estima parámetros: guarda los datos. Devuelve self."""
        self.X_ = np.asarray(X, dtype=float)
        y_arr = np.asarray(y)
        self.classes_ = np.unique(y_arr)
        self.y_codificada_ = np.searchsorted(self.classes_, y_arr)
        return self

    def _distancias(self, X):
        """Matriz (n_consultas, n_entrenamiento) con la distancia entre cada consulta y cada ejemplo.

        Pista para la euclídea: ||a - b||² = ||a||² + ||b||² - 2 a·b. Por redondeo puede dar valores
        levemente negativos: recortalos a 0 antes de la raíz cuadrada.
        """
        X_arr = np.asarray(X, dtype=float)
        if self.metrica == "euclidea":
            norm_q2 = np.sum(X_arr ** 2, axis=1, keepdims=True)
            norm_tr2 = np.sum(self.X_ ** 2, axis=1, keepdims=True).T
            dot = np.dot(X_arr, self.X_.T)
            dist2 = np.maximum(0.0, norm_q2 + norm_tr2 - 2.0 * dot)
            return np.sqrt(dist2)
        elif self.metrica == "manhattan":
            m = len(X_arr)
            n = len(self.X_)
            dist = np.zeros((m, n), dtype=float)
            for col in range(self.X_.shape[1]):
                dist += np.abs(X_arr[:, col:col+1] - self.X_[:, col])
            return dist
        else:
            raise ValueError(f"Métrica desconocida: {self.metrica}")

    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        """Devuelve (distancias, indices), ambos de forma (n_consultas, k), ordenados de menor a mayor distancia.

        Pista: np.argpartition encuentra los k menores sin ordenar toda la fila; después ordená solo esos k.
        """
        if X is None:
            X_arr = self.X_
        else:
            X_arr = np.asarray(X, dtype=float)

        k = self.k if n_neighbors is None else n_neighbors
        n_consultas = len(X_arr)
        n_train = len(self.X_)
        k = min(k, n_train)

        todas_dist = []
        todos_idx = []

        lote = max(1, self.tamano_lote)
        for i in range(0, n_consultas, lote):
            X_lote = X_arr[i:i + lote]
            D_lote = self._distancias(X_lote)

            if k < n_train:
                part_idx = np.argpartition(D_lote, kth=k - 1, axis=1)[:, :k]
                part_dist = np.take_along_axis(D_lote, part_idx, axis=1)

                orden = np.argsort(part_dist, axis=1)
                k_dist = np.take_along_axis(part_dist, orden, axis=1)
                k_idx = np.take_along_axis(part_idx, orden, axis=1)
            else:
                orden = np.argsort(D_lote, axis=1)
                k_dist = np.take_along_axis(D_lote, orden, axis=1)
                k_idx = orden

            todas_dist.append(k_dist)
            todos_idx.append(k_idx)

        res_dist = np.vstack(todas_dist)
        res_idx = np.vstack(todos_idx)

        if return_distance:
            return res_dist, res_idx
        return res_idx

    def predict_proba(self, X):
        """Proporción (ponderada según `pesos`) de vecinos de cada clase. Columnas en el orden de classes_."""
        X_arr = np.asarray(X, dtype=float)
        dist, idx = self.kneighbors(X_arr)

        n_consultas = len(X_arr)
        n_clases = len(self.classes_)
        vecinos_clases = self.y_codificada_[idx]

        if self.pesos == "uniforme":
            prob = np.zeros((n_consultas, n_clases), dtype=float)
            for c in range(n_clases):
                prob[:, c] = np.mean(vecinos_clases == c, axis=1)
            return prob

        elif self.pesos == "distancia":
            tiene_cero = (dist == 0.0)
            filas_con_cero = np.any(tiene_cero, axis=1)

            pesos = np.zeros_like(dist, dtype=float)

            if np.any(~filas_con_cero):
                pesos[~filas_con_cero] = 1.0 / dist[~filas_con_cero]

            if np.any(filas_con_cero):
                pesos[filas_con_cero] = tiene_cero[filas_con_cero].astype(float)

            pesos_suma = np.sum(pesos, axis=1, keepdims=True)
            pesos = pesos / pesos_suma

            prob = np.zeros((n_consultas, n_clases), dtype=float)
            for c in range(n_clases):
                prob[:, c] = np.sum(pesos * (vecinos_clases == c), axis=1)
            return prob
        else:
            raise ValueError(f"Pesos no reconocidos: {self.pesos}")

    def predict(self, X):
        """Clase con mayor probabilidad. En caso de empate, la primera en classes_ (como np.argmax)."""
        proba = self.predict_proba(X)
        indices_max = np.argmax(proba, axis=1)
        return self.classes_[indices_max]
