# Contexto para la siguiente sesión — House Prices

_Última actualización: 2026-10-29_

## 1. Objetivo
Predecir `SalePrice` del dataset **House Prices - Advanced Regression Techniques** (Kaggle) y disponibilizar el modelo ganador (próximamente con FastAPI).

## 2. Estado actual
- [x] Proyecto creado con **cookiecutter-data-science v2.3.0** (`proyecto_pd/`).
- [x] Datos en `data/raw/`: `train.csv` (1460 × 81), `test.csv` (1459 × 80), `data_description.txt`, `sample_submission.csv`.
- [x] Entorno conda `env_house_pricing` (Python 3.11).
- [x] EDA → `notebooks/1.0-de-analisis_exploratorio.ipynb` (sección 5).
- [x] 30 features en `pipeline_config.yaml` → `MODEL_FEATURES` (sección 6).
- [x] Ingeniería de características → `notebooks/2.0-de-ingenieria_caracteristicas.ipynb` (sección 8).
- [x] Torneo de modelos → `notebooks/3.0-de-entrenamiento_modelos.ipynb`; **ganador: GradientBoosting** (sección 9).
- [x] Inferencia → `notebooks/4.0-de-inferencia.ipynb` (sección 10).
- [x] Script de producción `scripts/inferencia.py` + pruebas `tests/test_inferencia.py` (25 pruebas OK) (sección 11).
- [ ] **Siguiente paso:** exponer `scripts/inferencia.py` con **FastAPI** (sección 12).

## 3. Entorno
| Elemento | Valor |
|---|---|
| Anaconda | `C:\Users\doty_\anaconda3` (`conda` **no** está en el PATH de PowerShell) |
| Entorno | `env_house_pricing` — Python 3.11 |
| Python del entorno | `C:\Users\doty_\anaconda3\envs\env_house_pricing\python.exe` |
| Activar | Anaconda Prompt → `conda activate env_house_pricing` |
| VS Code | Seleccionar kernel / intérprete `env_house_pricing` |
| Dependencias | `requirements.txt` (incluye `-e .` para el paquete local `proyecto_pd`) |

Librerías: numpy, pandas 3.1, matplotlib, seaborn, scikit-learn 1.9, feature-engine 1.9, pyyaml, python-dotenv, loguru, tqdm, pytest, ruff, ipykernel, proyecto_pd (editable).

> - `conda run` falla con scripts de varias líneas → usar directamente el `python.exe` del entorno.
> - `pyproject.toml`: `requires-python = "~=3.11.0"`, config de pytest (`pythonpath = ["."]`, `testpaths = ["tests"]`) y de ruff (incluye `scripts/` y `tests/`, docstrings estilo Google, `known-first-party = ["proyecto_pd", "scripts"]`).
> - `environment.yml` (generado por cookiecutter) **no** se usa.
> - feature-engine emite un `Pandas4Warning` (`inplace` en `drop`) con pandas 3.1: es solo una advertencia.

## 4. Convenciones y forma de trabajo acordadas
1. **Nombres de notebooks:** `<paso>-de-<descripcion>.ipynb` (`de` = Dorotthy Estrada). Siguiente fase: `5.x`.
2. **Celdas de notebook:** siempre con **título y descripción** de lo que hacen; cada paso en celdas separadas.
3. **Dependencias:** toda librería nueva se instala en `env_house_pricing` **y** se agrega a `requirements.txt`.
4. **Rutas:** en notebooks, constantes de `proyecto_pd.config`; nunca rutas fijas. Figuras en `reports/figures/`.
5. **No asumir:** ante cualquier decisión metodológica (métricas, modelos, transformaciones, divisiones, ubicaciones) **preguntar antes**; no hacer nada adicional a lo pedido.
6. **Código de producción:** PEP 8 (verificado con ruff), firmas con tipos en parámetros y retornos, docstrings estilo Google, `logging` estándar.

## 5. Resumen del análisis exploratorio (notebook 1.0)
| Sección | Contenido |
|---|---|
| 1–3 | Configuración, carga, estructura (36 numéricas, 43 categóricas) |
| 4 | Valores faltantes (en muchas variables NA = "no tiene") |
| 5 | Target: `SalePrice` asimetría 1.88, curtosis 6.54 |
| 6.1 | **Discretas** (≤ 15 valores distintos, excepto `PoolArea`) vs **continuas** |
| 6.2 | Scatter continuas vs target + `df_correlaciones_continuas` (r de Pearson) |
| 6.3 | Boxplot + stripplot de discretas vs target |
| 7.1 | Boxplot de categóricas (NA → `"Ausente"`) + `df_eta_categoricas` (razón de correlación η) |
| 8 | Selección top 10 por grupo → lista `features` |
| 9 | Conclusiones (**pendiente de completar**) |

Métricas: continuas → |r| de Pearson; discretas y categóricas → **η** = √(variación entre grupos / variación total), rango 0–1 (η² = R² de ANOVA de una vía; se infla con categorías pequeñas o numerosas).

## 6. Features seleccionadas (`pipeline_config.yaml` → `MODEL_FEATURES`)
| Continuas (\|r\|) | Discretas (η) | Categóricas (η) |
|---|---|---|
| GrLivArea (0.71) | OverallQual (0.83) | Neighborhood (0.74) |
| GarageArea (0.62) | GarageCars (0.70) | ExterQual (0.69) |
| TotalBsmtSF (0.61) | FullBath (0.58) | BsmtQual (0.68) |
| 1stFlrSF (0.61) | TotRmsAbvGrd (0.55) | KitchenQual (0.68) |
| YearBuilt (0.52) | MSSubClass (0.50) | GarageFinish (0.55) |
| YearRemodAdd (0.51) | Fireplaces (0.48) | FireplaceQu (0.54) — *descartada en FE* |
| GarageYrBlt (0.49) | OverallCond (0.35) | Foundation (0.51) |
| MasVnrArea (0.48) | HalfBath (0.30) | GarageType (0.50) |
| BsmtFinSF1 (0.39) | BedroomAbvGr (0.24) | BsmtFinType1 (0.46) |
| LotFrontage (0.35) | BsmtFullBath (0.23) | HeatingQC (0.44) |

## 7. Recomendaciones del EDA no aplicadas (por si se quiere mejorar el modelo)
- Variables nuevas: `TotalSF`, `edad = YrSold − YearBuilt`, `anios_desde_remodelacion`, `banos_totales`.
- Indicadores binarios `tiene_X` para variables con muchos ceros (2ndFlrSF, MasVnrArea, WoodDeckSF…).
- Quitar las 2 casas con `GrLivArea > 4000` y precio bajo (se optó por capping IQR).
- Pares redundantes (GarageCars/GarageArea, TotalBsmtSF/1stFlrSF, GrLivArea/TotRmsAbvGrd, YearBuilt/GarageYrBlt): elegir uno por par en modelos lineales.
- Target en log: en el torneo con log los modelos lineales mejoraron (Ridge 28,914 vs 34,463 en dólares); la usuaria eligió **entrenar en dólares**.

## 8. Ingeniería de características (notebook 2.0)
Pipeline sklearn con transformadores de feature-engine, ajustado solo con train.

| Paso | Decisión de la usuaria |
|---|---|
| División | `train.csv` 80/20 (`random_state=42`) → train 1168, validación 292 |
| Imputación numérica | % faltantes en train: ≤5% mediana (`MeanMedianImputer`); 5–20% muestreo aleatorio (`RandomSampleImputer`: GarageYrBlt, LotFrontage); >20% descartar |
| Muestreo aleatorio | `seed="observation"` con `VARS_SEMILLA = [GrLivArea, YearBuilt, OverallQual]` → la imputación de una casa no depende del lote (inferencia reproducible) |
| Imputación categórica | Moda (`CategoricalImputer`); regla >20% también aplica → **FireplaceQu descartada** (47%) → 29 features |
| Ordinales | Mapeo manual (`OrdinalEncoder` + `SklearnTransformerWrapper`): Po=1…Ex=5; GarageFinish Unf=1<RFn<Fin; BsmtFinType1 Unf=1<…<GLQ=6 |
| Nominales | Neighborhood, Foundation, GarageType, **MSSubClass (como nominal)**: `RareLabelEncoder` (tol 1%) + `OneHotEncoder` |
| Outliers | `Winsorizer` IQR fold 1.5, ambas colas, solo continuas (no se eliminan filas) |
| Transformación | log: LotFrontage, 1stFlrSF, GrLivArea · Yeo-Johnson: MasVnrArea, BsmtFinSF1, TotalBsmtSF |
| Escalado | `StandardScaler` en todas las columnas (incluidas one-hot) |

**Salidas:** `data/processed/train_fe.csv` (1168 × 73), `validacion_fe.csv` (292 × 73), `test_fe.csv` (1459 × 71) — 70 features + `Id` (+ `SalePrice`, `SalePrice_log`). Pipeline: `models/pipeline_fe.joblib`.

**Puntos a revisar con la usuaria (no se cambiaron):**
- El capping IQR sube a 41 pies² a las 28 casas sin sótano (TotalBsmtSF = 0) y recorta 77 valores de MasVnrArea (59% son ceros).
- LotFrontage queda casi simétrica tras el capping (skew 0.11); el log la sobrecorrige (−0.79).
- One-hot sin `drop_last` (sin problema para modelos regularizados o de árboles).

## 9. Entrenamiento y torneo de modelos (notebook 3.0)
- **Sin fuga:** se recrea el split 80/20 desde datos crudos (se verifican los `Id` contra `validacion_fe.csv`) y `Pipeline(clone(pipeline_fe) + modelo)` → el FE se reajusta **dentro de cada fold**.
- **CV:** `KFold(n_splits=10, shuffle=True, random_state=42)` + `GridSearchCV` (`neg_root_mean_squared_error`, `return_train_score=True`), 4–9 configuraciones por modelo.
- **Target: `SalePrice` en dólares** (la usuaria lo cambió desde log).
- **Torneo en validación:** RMSE real (USD) y NRMSE = RMSE / (máx − mín de `SalePrice` en validación = 719,689).
- **Gráficas:** real vs predicho + residuos por modelo (`reports/figures/modelos_real_vs_predicho.png`) y barras del torneo (`modelos_torneo.png`).

| # | Modelo | Mejores hiperparámetros | RMSE CV | Brecha CV − train | RMSE validación | NRMSE |
|---|---|---|---|---|---|---|
| 1 | **GradientBoosting** | `learning_rate=0.03`, `max_depth=3` (500 árboles, `subsample=0.8`) | 26,223 | 13,829 | **26,577** | **0.0369** |
| 2 | RandomForest | `max_depth=None`, `max_features=0.5`, `min_samples_leaf=1` | 27,710 | 16,714 | 27,903 | 0.0388 |
| 3 | Lasso | `alpha=300` | 30,886 | 1,420 | 34,229 | 0.0476 |
| 4 | ElasticNet | `alpha=0.1`, `l1_ratio=0.2` | 30,891 | 1,383 | 34,448 | 0.0479 |
| 5 | Ridge | `alpha=100` | 30,887 | 1,333 | 34,463 | 0.0479 |
| 6 | MLP | `alpha=0.01`, `hidden_layer_sizes=(128, 64)`, early stopping | 32,198 | 5,778 | 37,219 | 0.0517 |

Observaciones:
- Los modelos lineales subestiman casas caras (residuos curvos por encima de ~400k).
- Hiperparámetros en el borde de la rejilla (Ridge `alpha=100` máx, GB `learning_rate=0.03` mín, MLP red más grande): ampliar rejillas podría mejorar.
- **Modelo ganador guardado** (sección 9 del notebook): GradientBoosting **reentrenado con train + validación (1460 filas)** → `models/modelo_ganador.joblib` (FE + modelo).

## 10. Inferencia (notebook 4.0)
- Entrada: `test.csv` **crudo** (80 columnas).
- `models/pipeline_inferencia.joblib` = `MatchVariables` (selecciona/ordena las 30 features; agrega como NaN las faltantes) → `features` (pipeline FE) → `modelo` (GradientBoosting).
- Salida: `data/processed/predicciones_test.csv` en formato Kaggle (`Id`, `SalePrice`), 1459 filas, verificado contra `sample_submission.csv`.
- Uso: `joblib.load("models/pipeline_inferencia.joblib").predict(datos_crudos)`.

## 11. Script de producción y pruebas
**`scripts/inferencia.py`** (PEP 8 verificado con ruff, tipos y docstrings):

| Función | Firma |
|---|---|
| `predecir` | `(datos: pd.DataFrame, pipeline: Optional[Pipeline] = None) -> pd.DataFrame` → columnas `Id`, `SalePrice`, `prediccion_valida` |
| `predecir_registros` | `(registros: list[dict[str, Any]], pipeline=None) -> list[dict[str, Any]]` → **interfaz para FastAPI** (JSON) |
| `cargar_pipeline` | `(ruta: Path = RUTA_PIPELINE) -> Pipeline` (cacheado con `lru_cache`) |
| `validar_entrada` | `(datos, pipeline) -> pd.DataFrame` |
| `evaluar_predicciones` | `(predicciones: np.ndarray) -> np.ndarray` |
| `configurar_logging` | `(carpeta: Path = CARPETA_SCRIPTS, nivel: int = logging.INFO) -> Path` |

- **Errores:** `EntradaInferenciaError` (entrada no DataFrame/lista, vacía, faltan columnas, texto en numéricas) y `PrediccionError` (falla del modelo o filas de salida ≠ entrada).
- **Calidad:** predicción coherente = finita y en `[PRECIO_MINIMO, PRECIO_MAXIMO]` = [10,000, 2,000,000] USD; las incoherentes se devuelven marcadas y se registra un ERROR por `Id`. WARNING por faltantes a imputar e `Id` duplicados.
- **Logging:** módulo `logging` estándar → `scripts/inferencia_AAAAMMDD_HHMMSS.log` (uno por proceso) + consola; resumen por ejecución (entrada, salida, coherentes, incoherentes, duración, min/mediana/max). `*.log` está en `.gitignore`.

**`tests/test_inferencia.py`** (pytest, **25 pruebas OK**): muestras aleatorias de 25 filas de `test.csv` con 5 semillas; coincidencia con el notebook 4.0; la predicción no depende del lote; interfaz JSON; columnas extra/desordenadas; faltantes; cada tipo de error; predicciones incoherentes marcadas; log con estampa de tiempo.

Comandos (desde `proyecto_pd/`):
```
C:\Users\doty_\anaconda3\envs\env_house_pricing\python.exe -m pytest
C:\Users\doty_\anaconda3\envs\env_house_pricing\python.exe -m ruff check scripts tests --select E,W,F,I,N,D
C:\Users\doty_\anaconda3\envs\env_house_pricing\python.exe -m ruff format --check scripts tests
```

## 12. Próximos pasos sugeridos (confirmar con la usuaria antes de decidir)
1. **API con FastAPI** sobre `predecir_registros` (instalar `fastapi` + `uvicorn` y agregarlos a `requirements.txt`). Definir con la usuaria: ubicación del archivo de la API, endpoints (p. ej. `/predict`, `/health`), esquema de entrada (pydantic) y manejo de errores → `EntradaInferenciaError` = 422/400, `PrediccionError` = 500.
2. Pendientes opcionales: completar las conclusiones del EDA (sección 9 del notebook 1.0); revisar los puntos de la sección 8; ampliar rejillas en el borde; probar variables nuevas de la sección 7.

## 13. Archivos clave
| Ruta | Qué es |
|---|---|
| `pipeline_config.yaml` | `MODEL_FEATURES` (30 features) |
| `notebooks/1.0` … `4.0` | EDA, FE, entrenamiento, inferencia |
| `data/processed/` | `train_fe.csv`, `validacion_fe.csv`, `test_fe.csv`, `predicciones_test.csv` |
| `models/pipeline_fe.joblib` | Pipeline de FE ajustado con train (80%) |
| `models/modelo_ganador.joblib` | FE + GradientBoosting, ajustado con train + validación |
| `models/pipeline_inferencia.joblib` | Selección de features + FE + modelo (datos crudos → predicción) |
| `scripts/inferencia.py` | Script de inferencia para producción / FastAPI |
| `tests/test_inferencia.py` | Pruebas pytest del script |
| `reports/figures/` | Gráficas del EDA, FE y modelos |
