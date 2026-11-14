# Ficha CRISP-ML(Q) · Churn Predictor

CRISP-ML(Q) organiza un proyecto de ML en seis fases. La Q (quality assurance) pide que
cada fase declare sus requisitos, los riesgos que pueden hacerla fallar y el control que
los detecta. Esta ficha es el registro de esas decisiones: se versiona junto al código.

## Fase 1 · Negocio y datos

| Pregunta | Respuesta |
|---|---|
| Problema de negocio | La fintech pierde clientes y hoy contacta a todos por igual en las campañas de retención |
| Decisión que apoya el modelo | A qué clientes ofrecer retención este mes |
| Criterio de éxito de negocio | Gastar menos por cliente que la política actual (contactar a todos) |
| Criterio de éxito de ML | F1 ≥ 0.60 y ROC-AUC ≥ 0.70 en test |
| Criterio de éxito económico | Costo esperado por cliente menor que contactar a todos y que no contactar a nadie |
| Costo de un falso negativo | S/ 200 (cliente que se va sin oferta) |
| Costo de un falso positivo | S/ 100 (oferta a quien no se iba) |
| Factibilidad | 300 clientes, 7 features, clase balanceada (49.7 % churn) |

## Riesgos y controles por fase

| Fase | Riesgo | Control (quality gate) | Dónde vive |
|---|---|---|---|
| 1. Negocio y datos | El dataset cambia de esquema o trae valores imposibles | Contrato de datos | `src/validate_data.py`, job `data-validation` |
| 1. Negocio y datos | Mismo cliente cargado dos veces con otro id | Check `no_full_duplicates` + test | `src/validate_data.py`, `tests/test_data_contract.py` |
| 1. Negocio y datos | `"None"` leído como nulo y luego imputado | `keep_default_na=False` + test | `src/common.py`, `tests/test_data_contract.py` |
| 2. Preparación | Training-serving skew: la API preprocesa distinto que el entrenamiento | Preprocesamiento dentro del Pipeline | `src/features.py`, `tests/test_features.py` |
| 2. Preparación | Data leakage: el escalador ve datos de test | El Pipeline se ajusta solo con train, fold por fold | `src/train.py` |
| 3. Modelado | Elegir modelo mirando test (optimismo) | Selección por validación cruzada | `src/train.py` |
| 3. Modelado | Un modelo que no aporta sobre lo trivial | Baseline registrado en MLflow | `src/train.py` |
| 4. Evaluación | Buena métrica promedio, mala en un segmento | Recall por segmento, brecha máxima | `src/evaluate_and_register.py` |
| 4. Evaluación | Promover un modelo peor que el vigente | Champion / challenger | `src/evaluate_and_register.py` |
| 5. Despliegue | La API sirve algo distinto a lo evaluado | ✅ Test de skew contra la API en ejecución (local, contenedor, staging y prod) | `tests/test_skew.py`, jobs `smoke-test`, `deploy-staging`, `deploy-production` |
| 5. Despliegue | Entradas inválidas llegan al modelo | ✅ Contrato en el esquema de la API (422) | `app/main.py`, `tests/test_api.py` |
| 5. Despliegue | La imagen nueva no arranca en el entorno | ✅ Health check antes de publicar + circuit breaker de ECS con rollback | job `build-and-publish`, `infra/ecs.tf`, `scripts/deploy_ecs.sh` |
| 5. Despliegue | Un cambio llega a clientes sin probarse | ✅ Staging desde `develop`, producción desde `main` con aprobación manual | `.github/workflows/mlops.yml`, Environment `production` |
| 5. Despliegue | Producción corre una imagen distinta a la probada | ✅ Promoción de la misma imagen de staging (sin rebuild en `main`) | job `build-and-publish` (reto 8) |
| 6. Monitoreo | Los datos de producción cambian | ✅ PSI + KS por feature, PSI de predicciones; alerta `ChurnModelPsiCritical` | `src/monitor.py`, `monitoring/prometheus/alerts.yml` |
| 6. Monitoreo | El modelo se degrada sin que cambien los datos | ✅ F1 con etiquetas tardías | `src/monitor.py`, `model-monitoring.yml` |
| 6. Monitoreo | Nadie actúa cuando el monitor decide reentrenar | ✅ Exit 42 dispara `mlops.yml` (Continuous Training) y abre un issue | `model-monitoring.yml` |
| 6. Monitoreo | Un champion nuevo predice mal en producción | ✅ Rollback moviendo el alias `@champion` o desplegando el tag del commit anterior | MLflow Registry, `scripts/deploy_ecs.sh` |

✅ = control implementado y verificado en el laboratorio 14.2.

### Riesgos que esta solución todavía no cubre

| Fase | Riesgo | Por qué no está cubierto | Control propuesto |
|---|---|---|---|
| 6. Monitoreo | El monitoreo en producción no tiene datos: `logs/predictions.jsonl` vive dentro del contenedor de ECS y se pierde cada vez que la tarea se reemplaza | `model-monitoring.yml` corre sobre escenarios simulados, no sobre el tráfico real de prod | Enviar el log de predicciones a S3 o CloudWatch Logs y que el monitor lea de ahí, junto con las etiquetas reales cuando lleguen |
| 5. Despliegue | La API es pública por HTTP, sin autenticación ni límite de peticiones | El ALB escucha en el puerto 80 abierto a internet | HTTPS con certificado de ACM, API key o autenticación, y WAF con rate limiting |

## Decisiones registradas

| Fecha | Decisión | Motivo |
|---|---|---|
| 2026-11-11 | Umbral de decisión elegido por costo, no 0.5 (umbral resultante 0.3) | Un FN cuesta el doble que un FP |
| 2026-11-11 | El modelo se selecciona por F1 de validación cruzada | Test queda reservado para la evaluación final |
| 2026-11-11 | Preprocesamiento dentro de un `Pipeline` de scikit-learn | Elimina el training-serving skew (la API ignoraba `contract_type`) |
| 2026-11-11 | Se agrega `gradient_boosting` como cuarto candidato; el elegido sigue siendo `random_forest` | F1 CV 0.6476 ± 0.0761 y ROC-AUC CV 0.6858, por debajo de random_forest (0.6900 / 0.7326) |
| 2026-11-11 | Se mantienen los costos FN 200 / FP 100 | Con FN 500 / FP 50 el umbral óptimo es 0.05 (contactar a todos) y el modelo no ahorra frente a esa política |
| 2026-11-11 | `contract_type` no entra al gate de segmentos; se reporta aparte | Brecha de recall 0.50 (> 0.25) por pocos churners en contratos largos; no es un atributo sensible |
| 2026-11-13 | Un entorno por rama: `develop` → staging, `main` → producción con aprobación | El modelo se prueba en infraestructura real antes de llegar a clientes |
| 2026-11-13 | `main` promueve la imagen de staging en lugar de reconstruirla (reto 8) | Producción corre exactamente el artefacto que se probó |
| 2026-11-13 | El monitor reentrena con 2+ features en PSI crítico, PSI de predicción crítico o caída de F1 ≥ 0.10 | Las señales sin etiquetas avisan antes; el F1 confirma el concept drift |
| 2026-11-11 | Contrato de datos con check de filas duplicadas completas | Un cliente duplicado con otro id sesga el entrenamiento y no lo detecta `unique_id` |
