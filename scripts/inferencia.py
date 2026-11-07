"""Inferencia del modelo ganador de House Prices.

Expone funciones para predecir ``SalePrice`` a partir de datos crudos (mismas
columnas que ``test.csv``) usando el pipeline completo generado en el proceso
de investigación y desarrollo (``models/pipeline_inferencia.joblib``):
selección de features -> ingeniería de características -> modelo ganador.

Pensado para ser consumido por una API (FastAPI):

- :func:`predecir` recibe y devuelve ``pandas.DataFrame``.
- :func:`predecir_registros` recibe y devuelve listas de diccionarios (JSON).

Cada ejecución registra su actividad en un archivo
``inferencia_<AAAAMMDD_HHMMSS>.log`` dentro de esta misma carpeta.
"""

from datetime import datetime
from functools import lru_cache
import logging
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

CARPETA_SCRIPTS = Path(__file__).resolve().parent
RAIZ_PROYECTO = CARPETA_SCRIPTS.parent
RUTA_PIPELINE = RAIZ_PROYECTO / "models" / "pipeline_inferencia.joblib"

COLUMNA_ID = "Id"
COLUMNA_PREDICCION = "SalePrice"
COLUMNA_VALIDA = "prediccion_valida"

# Rango de precios considerado coherente (USD). En train SalePrice va de
# 34,900 a 755,000; se deja margen para casas fuera de ese rango.
PRECIO_MINIMO = 10_000.0
PRECIO_MAXIMO = 2_000_000.0

NOMBRE_LOGGER = "inferencia"
FORMATO_LOG = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

logger = logging.getLogger(NOMBRE_LOGGER)


class EntradaInferenciaError(ValueError):
    """Los datos de entrada no cumplen el contrato esperado por el modelo."""


class PrediccionError(RuntimeError):
    """La salida del modelo no es consistente con la entrada."""


def configurar_logging(carpeta: Path = CARPETA_SCRIPTS, nivel: int = logging.INFO) -> Path:
    """Configura el logger de inferencia con un archivo con estampa de tiempo.

    Solo se configura una vez por proceso: si el logger ya tiene un archivo
    asociado, se reutiliza.

    Args:
        carpeta: Carpeta donde se crea el archivo de log.
        nivel: Nivel mínimo de los mensajes a registrar.

    Returns:
        Ruta del archivo de log en uso.
    """
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            return Path(handler.baseFilename)

    estampa = datetime.now().strftime("%Y%m%d_%H%M%S")
    ruta_log = carpeta / f"inferencia_{estampa}.log"
    formato = logging.Formatter(FORMATO_LOG)

    manejador_archivo = logging.FileHandler(ruta_log, encoding="utf-8")
    manejador_archivo.setFormatter(formato)
    manejador_consola = logging.StreamHandler()
    manejador_consola.setFormatter(formato)

    logger.setLevel(nivel)
    logger.addHandler(manejador_archivo)
    logger.addHandler(manejador_consola)
    logger.propagate = False
    logger.info("Log de inferencia iniciado en %s", ruta_log)
    return ruta_log


@lru_cache(maxsize=1)
def cargar_pipeline(ruta: Path = RUTA_PIPELINE) -> Pipeline:
    """Carga el pipeline de inferencia (se cachea tras la primera carga).

    Args:
        ruta: Ruta del archivo ``.joblib`` con el pipeline completo.

    Returns:
        Pipeline ajustado: selección de features, ingeniería de
        características y modelo ganador.

    Raises:
        FileNotFoundError: Si el archivo del pipeline no existe.
    """
    configurar_logging()
    if not ruta.exists():
        logger.error("No se encontró el pipeline en %s", ruta)
        raise FileNotFoundError(f"No se encontró el pipeline en {ruta}")

    pipeline = joblib.load(ruta)
    logger.info(
        "Pipeline cargado desde %s | pasos: %s",
        ruta,
        [nombre for nombre, _ in pipeline.steps],
    )
    return pipeline


def obtener_features(pipeline: Pipeline) -> list[str]:
    """Devuelve las features crudas que necesita el pipeline.

    Args:
        pipeline: Pipeline de inferencia ajustado.

    Returns:
        Nombres de las columnas de entrada, en el orden de entrenamiento.
    """
    return list(pipeline.named_steps["seleccionar_features"].feature_names_in_)


def obtener_features_numericas(pipeline: Pipeline) -> list[str]:
    """Devuelve las features crudas que deben ser numéricas.

    Se toman de los imputadores numéricos del pipeline de ingeniería de
    características (mediana y muestreo aleatorio).

    Args:
        pipeline: Pipeline de inferencia ajustado.

    Returns:
        Nombres de las columnas numéricas de entrada.
    """
    pasos_fe = pipeline.named_steps["features"].named_steps
    return list(pasos_fe["imputar_mediana"].variables_) + list(
        pasos_fe["imputar_aleatorio"].variables_
    )


def validar_entrada(datos: pd.DataFrame, pipeline: Pipeline) -> pd.DataFrame:
    """Valida los datos de entrada y los prepara para el pipeline.

    Reglas:

    - Debe ser un ``DataFrame`` con al menos una fila (error).
    - Deben estar todas las columnas de features del modelo (error).
    - Las columnas numéricas deben poder convertirse a número (error).
    - ``Id`` duplicados y valores faltantes se permiten, pero se registran
      como advertencia (los faltantes los imputa el pipeline).

    Args:
        datos: Datos crudos con las columnas de ``test.csv``.
        pipeline: Pipeline de inferencia ajustado.

    Returns:
        Copia de los datos indexada por ``Id`` (si existe la columna) y con
        las columnas numéricas convertidas a número.

    Raises:
        EntradaInferenciaError: Si la entrada no cumple alguna regla.
    """
    if not isinstance(datos, pd.DataFrame):
        mensaje = f"Se esperaba un pandas.DataFrame y se recibió {type(datos).__name__}"
        logger.error(mensaje)
        raise EntradaInferenciaError(mensaje)

    if datos.empty:
        mensaje = "La entrada no tiene filas"
        logger.error(mensaje)
        raise EntradaInferenciaError(mensaje)

    features = obtener_features(pipeline)
    faltantes = [col for col in features if col not in datos.columns]
    if faltantes:
        mensaje = f"Faltan {len(faltantes)} columnas requeridas: {faltantes}"
        logger.error(mensaje)
        raise EntradaInferenciaError(mensaje)

    preparados = datos.copy()
    if COLUMNA_ID in preparados.columns:
        preparados = preparados.set_index(COLUMNA_ID)

    no_numericas = []
    for col in obtener_features_numericas(pipeline):
        try:
            preparados[col] = pd.to_numeric(preparados[col])
        except (TypeError, ValueError):
            no_numericas.append(col)
    if no_numericas:
        mensaje = f"Columnas que deben ser numéricas con valores no numéricos: {no_numericas}"
        logger.error(mensaje)
        raise EntradaInferenciaError(mensaje)

    ids_duplicados = preparados.index[preparados.index.duplicated()].unique().tolist()
    if ids_duplicados:
        logger.warning("Id duplicados en la entrada: %s", ids_duplicados)

    descartadas = pipeline.named_steps["features"].named_steps["descartar_faltantes"]
    usadas = [col for col in features if col not in descartadas.features_to_drop]
    nulos = preparados[usadas].isna().sum()
    nulos = nulos[nulos > 0]
    if not nulos.empty:
        logger.warning("Valores faltantes que imputará el pipeline: %s", nulos.to_dict())

    logger.info(
        "Entrada válida: %d filas, %d columnas (%d features usadas)",
        len(preparados),
        datos.shape[1],
        len(features),
    )
    return preparados


def evaluar_predicciones(predicciones: np.ndarray) -> np.ndarray:
    """Marca qué predicciones son coherentes.

    Una predicción es coherente si es un número finito dentro del rango
    ``[PRECIO_MINIMO, PRECIO_MAXIMO]``.

    Args:
        predicciones: Predicciones del modelo en USD.

    Returns:
        Arreglo booleano: ``True`` si la predicción es coherente.
    """
    valores = np.asarray(predicciones, dtype=float)
    return np.isfinite(valores) & (valores >= PRECIO_MINIMO) & (valores <= PRECIO_MAXIMO)


def predecir(datos: pd.DataFrame, pipeline: Optional[Pipeline] = None) -> pd.DataFrame:
    """Genera las predicciones de ``SalePrice`` a partir de datos crudos.

    Controles de calidad entre entrada y salida:

    - Valida la entrada (ver :func:`validar_entrada`).
    - Verifica que haya exactamente una predicción por fila de entrada; si no,
      lanza :class:`PrediccionError`.
    - Cuenta las predicciones coherentes e incoherentes (ver
      :func:`evaluar_predicciones`) y registra un error por cada fila
      incoherente. Esas filas se devuelven marcadas, no se descartan.

    Args:
        datos: Datos crudos con las columnas de ``test.csv``. Si incluye la
            columna ``Id`` se usa como identificador; si no, se usa el índice.
        pipeline: Pipeline de inferencia. Si es ``None`` se carga el de
            ``models/pipeline_inferencia.joblib``.

    Returns:
        DataFrame con las columnas ``Id``, ``SalePrice`` (USD) y
        ``prediccion_valida``.

    Raises:
        EntradaInferenciaError: Si la entrada no es válida.
        PrediccionError: Si el modelo falla o la salida no corresponde a la
            entrada.
    """
    configurar_logging()
    if pipeline is None:
        pipeline = cargar_pipeline()

    inicio = datetime.now()
    logger.info("Inicio de predicción: %d filas recibidas", len(datos))
    preparados = validar_entrada(datos, pipeline)

    try:
        valores = np.asarray(pipeline.predict(preparados), dtype=float)
    except Exception as error:
        logger.exception("Error del pipeline al predecir")
        raise PrediccionError(f"Error del pipeline al predecir: {error}") from error

    if valores.shape != (len(preparados),):
        mensaje = (
            f"Filas de salida ({valores.size}) distintas a filas de entrada ({len(preparados)})"
        )
        logger.error(mensaje)
        raise PrediccionError(mensaje)

    validas = evaluar_predicciones(valores)
    resultado = pd.DataFrame(
        {
            COLUMNA_ID: preparados.index,
            COLUMNA_PREDICCION: valores,
            COLUMNA_VALIDA: validas,
        }
    )

    for _, fila in resultado.loc[~validas].iterrows():
        logger.error(
            "Predicción incoherente | Id=%s | SalePrice=%s | rango esperado [%s, %s]",
            fila[COLUMNA_ID],
            fila[COLUMNA_PREDICCION],
            PRECIO_MINIMO,
            PRECIO_MAXIMO,
        )

    n_validas = int(validas.sum())
    duracion = (datetime.now() - inicio).total_seconds()
    logger.info(
        "Fin de predicción | entrada: %d | salida: %d | coherentes: %d | "
        "incoherentes: %d | duración: %.3f s",
        len(preparados),
        len(resultado),
        n_validas,
        len(resultado) - n_validas,
        duracion,
    )
    if n_validas:
        logger.info(
            "Resumen SalePrice coherente | min: %.0f | mediana: %.0f | max: %.0f",
            resultado.loc[validas, COLUMNA_PREDICCION].min(),
            resultado.loc[validas, COLUMNA_PREDICCION].median(),
            resultado.loc[validas, COLUMNA_PREDICCION].max(),
        )
    return resultado


def predecir_registros(
    registros: list[dict[str, Any]], pipeline: Optional[Pipeline] = None
) -> list[dict[str, Any]]:
    """Genera predicciones a partir de registros tipo JSON.

    Es la interfaz pensada para FastAPI: cada registro es un diccionario con
    las columnas de ``test.csv`` (los valores faltantes pueden venir como
    ``None``).

    Args:
        registros: Lista de casas a predecir.
        pipeline: Pipeline de inferencia. Si es ``None`` se carga el de
            ``models/pipeline_inferencia.joblib``.

    Returns:
        Lista de diccionarios con ``Id``, ``SalePrice`` y
        ``prediccion_valida``.

    Raises:
        EntradaInferenciaError: Si la entrada no es válida.
        PrediccionError: Si el modelo falla o la salida no corresponde a la
            entrada.
    """
    configurar_logging()
    if not isinstance(registros, list):
        mensaje = f"Se esperaba una lista de registros y se recibió {type(registros).__name__}"
        logger.error(mensaje)
        raise EntradaInferenciaError(mensaje)

    resultado = predecir(pd.DataFrame.from_records(registros), pipeline=pipeline)
    return resultado.to_dict(orient="records")
