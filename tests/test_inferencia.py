"""Pruebas del script de inferencia con observaciones aleatorias de test.csv."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from scripts import inferencia

RAIZ_PROYECTO = Path(__file__).resolve().parents[1]
RUTA_TEST = RAIZ_PROYECTO / "data" / "raw" / "test.csv"
RUTA_PREDICCIONES_NOTEBOOK = RAIZ_PROYECTO / "data" / "processed" / "predicciones_test.csv"

SEMILLAS = [0, 7, 42, 123, 2026]
TAMANIO_MUESTRA = 25


@pytest.fixture(scope="module")
def datos_test() -> pd.DataFrame:
    """Datos crudos completos de test.csv (80 columnas + Id)."""
    return pd.read_csv(RUTA_TEST)


@pytest.fixture(scope="module")
def pipeline() -> Pipeline:
    """Pipeline de inferencia del modelo ganador."""
    return inferencia.cargar_pipeline()


@pytest.fixture(params=SEMILLAS, ids=lambda semilla: f"semilla={semilla}")
def muestra_aleatoria(request: pytest.FixtureRequest, datos_test: pd.DataFrame) -> pd.DataFrame:
    """Observaciones aleatorias de test.csv, una muestra por semilla."""
    return datos_test.sample(n=TAMANIO_MUESTRA, random_state=request.param)


class PipelineFalso:
    """Imita al pipeline real pero devuelve predicciones fijas."""

    def __init__(self, pipeline_real: Pipeline, valores: list[float]) -> None:
        """Toma los pasos del pipeline real y los valores a devolver."""
        self.named_steps = pipeline_real.named_steps
        self.valores = valores

    def predict(self, datos: pd.DataFrame) -> np.ndarray:
        """Devuelve los valores fijos, sin importar la entrada."""
        return np.asarray(self.valores, dtype=float)


def test_predice_muestra_aleatoria(muestra_aleatoria: pd.DataFrame, pipeline: Pipeline) -> None:
    """Una predicción coherente por cada fila aleatoria, con sus mismos Id."""
    resultado = inferencia.predecir(muestra_aleatoria, pipeline=pipeline)

    assert list(resultado.columns) == ["Id", "SalePrice", "prediccion_valida"]
    assert len(resultado) == len(muestra_aleatoria)
    assert resultado["Id"].tolist() == muestra_aleatoria["Id"].tolist()
    assert resultado["prediccion_valida"].all()
    assert resultado["SalePrice"].between(inferencia.PRECIO_MINIMO, inferencia.PRECIO_MAXIMO).all()


def test_coincide_con_predicciones_del_notebook(
    muestra_aleatoria: pd.DataFrame, pipeline: Pipeline
) -> None:
    """El script reproduce las predicciones del notebook 4.0-de-inferencia."""
    esperado = pd.read_csv(RUTA_PREDICCIONES_NOTEBOOK, index_col="Id")
    resultado = inferencia.predecir(muestra_aleatoria, pipeline=pipeline).set_index("Id")

    np.testing.assert_allclose(resultado["SalePrice"], esperado.loc[resultado.index, "SalePrice"])


def test_prediccion_no_depende_del_lote(datos_test: pd.DataFrame, pipeline: Pipeline) -> None:
    """Una casa con faltantes imputados al azar da lo mismo sola que en un lote."""
    vars_aleatorias = pipeline.named_steps["features"].named_steps["imputar_aleatorio"].variables_
    con_faltantes = datos_test[datos_test[vars_aleatorias].isna().any(axis=1)]
    lote = con_faltantes.sample(n=TAMANIO_MUESTRA, random_state=5)

    en_lote = inferencia.predecir(lote, pipeline=pipeline)["SalePrice"].to_numpy()
    una_a_una = [
        inferencia.predecir(lote.iloc[[i]], pipeline=pipeline)["SalePrice"].item()
        for i in range(len(lote))
    ]

    np.testing.assert_allclose(en_lote, una_a_una)


def test_predecir_registros_json(muestra_aleatoria: pd.DataFrame, pipeline: Pipeline) -> None:
    """Interfaz JSON para FastAPI: faltantes como None y salida serializable."""
    registros = json.loads(muestra_aleatoria.to_json(orient="records"))

    salida = inferencia.predecir_registros(registros, pipeline=pipeline)

    assert len(salida) == len(registros)
    assert [fila["Id"] for fila in salida] == [registro["Id"] for registro in registros]
    assert all(fila["prediccion_valida"] for fila in salida)
    json.dumps(salida)


def test_columnas_extra_y_desordenadas(datos_test: pd.DataFrame, pipeline: Pipeline) -> None:
    """Columnas en otro orden o adicionales no cambian la predicción."""
    muestra = datos_test.sample(n=TAMANIO_MUESTRA, random_state=1)
    alterada = muestra[muestra.columns[::-1]].assign(columna_extra="x")

    original = inferencia.predecir(muestra, pipeline=pipeline)
    modificada = inferencia.predecir(alterada, pipeline=pipeline)

    np.testing.assert_allclose(original["SalePrice"], modificada["SalePrice"])


def test_faltantes_se_imputan(datos_test: pd.DataFrame, pipeline: Pipeline) -> None:
    """Valores faltantes en las features no impiden predecir."""
    muestra = datos_test.sample(n=TAMANIO_MUESTRA, random_state=2).copy()
    muestra.loc[muestra.index[:5], ["LotFrontage", "GarageType", "KitchenQual"]] = np.nan

    resultado = inferencia.predecir(muestra, pipeline=pipeline)

    assert resultado["prediccion_valida"].all()


def test_entrada_vacia_lanza_error(datos_test: pd.DataFrame, pipeline: Pipeline) -> None:
    """Una entrada sin filas es un error."""
    with pytest.raises(inferencia.EntradaInferenciaError, match="no tiene filas"):
        inferencia.predecir(datos_test.head(0), pipeline=pipeline)


def test_entrada_no_dataframe_lanza_error(pipeline: Pipeline) -> None:
    """Solo se aceptan DataFrames en predecir y listas en predecir_registros."""
    with pytest.raises(inferencia.EntradaInferenciaError, match="DataFrame"):
        inferencia.predecir([{"Id": 1}], pipeline=pipeline)
    with pytest.raises(inferencia.EntradaInferenciaError, match="lista"):
        inferencia.predecir_registros({"Id": 1}, pipeline=pipeline)


def test_columnas_faltantes_lanza_error(datos_test: pd.DataFrame, pipeline: Pipeline) -> None:
    """Si falta una feature del modelo, se lanza error indicando cuál."""
    muestra = datos_test.sample(n=TAMANIO_MUESTRA, random_state=3).drop(columns=["GrLivArea"])

    with pytest.raises(inferencia.EntradaInferenciaError, match="GrLivArea"):
        inferencia.predecir(muestra, pipeline=pipeline)


def test_valor_no_numerico_lanza_error(datos_test: pd.DataFrame, pipeline: Pipeline) -> None:
    """Texto en una feature numérica es un error."""
    muestra = datos_test.sample(n=TAMANIO_MUESTRA, random_state=4).astype({"GarageArea": object})
    muestra.iloc[0, muestra.columns.get_loc("GarageArea")] = "grande"

    with pytest.raises(inferencia.EntradaInferenciaError, match="GarageArea"):
        inferencia.predecir(muestra, pipeline=pipeline)


def test_predicciones_incoherentes_se_marcan(
    datos_test: pd.DataFrame, pipeline: Pipeline, caplog: pytest.LogCaptureFixture
) -> None:
    """Predicciones fuera de rango, negativas o NaN se marcan y se registran como error."""
    muestra = datos_test.head(4)
    falso = PipelineFalso(pipeline, [150_000.0, -5.0, np.nan, 9_000_000.0])
    inferencia.logger.propagate = True

    with caplog.at_level("ERROR", logger=inferencia.NOMBRE_LOGGER):
        resultado = inferencia.predecir(muestra, pipeline=falso)

    inferencia.logger.propagate = False
    assert resultado["prediccion_valida"].tolist() == [True, False, False, False]
    incoherentes = {mensaje for mensaje in caplog.messages if "Predicción incoherente" in mensaje}
    assert len(incoherentes) == 3


def test_salida_distinta_a_entrada_lanza_error(
    datos_test: pd.DataFrame, pipeline: Pipeline
) -> None:
    """Si el modelo devuelve otra cantidad de filas, se lanza error."""
    falso = PipelineFalso(pipeline, [150_000.0])

    with pytest.raises(inferencia.PrediccionError, match="distintas"):
        inferencia.predecir(datos_test.head(3), pipeline=falso)


def test_log_con_estampa_de_tiempo(datos_test: pd.DataFrame, pipeline: Pipeline) -> None:
    """El log se guarda en la carpeta scripts con estampa de tiempo y el resumen."""
    inferencia.predecir(datos_test.head(5), pipeline=pipeline)
    ruta_log = inferencia.configurar_logging()

    assert ruta_log.parent == inferencia.CARPETA_SCRIPTS
    assert ruta_log.name.startswith("inferencia_") and ruta_log.suffix == ".log"
    contenido = ruta_log.read_text(encoding="utf-8")
    assert "Fin de predicción | entrada: 5 | salida: 5" in contenido
