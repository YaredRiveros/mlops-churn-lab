# MLOps Churn Lab

Punto de partida de los laboratorios 14.1 y 14.2: predicción de churn para una fintech siguiendo CRISP-ML(Q).

El proyecto funciona, sus tests pasan y tiene un bug que solo aparece en producción. Encontrarlo y eliminarlo es el primer paso del laboratorio.

## El caso

**Predicción de Churn para una Fintech**

Una fintech detectó un incremento en la pérdida de clientes (churn). El área de analítica ya construyó una primera solución: entrena un modelo, registra experimentos en MLflow, lo publica como API y tiene un pipeline en GitHub Actions. Todo corre y los tests pasan, pero el negocio desconfía de las predicciones y nadie puede explicar por qué se eligió ese modelo ni cuándo habría que reemplazarlo.

El reto del equipo es llevar esa solución a un nivel confiable siguiendo CRISP-ML(Q):

- Validar los datos antes de entrenar.
- Garantizar que la API prediga lo mismo que el modelo evaluado.
- Elegir el modelo con evidencia y decidir con criterios de negocio.
- Promover a producción solo lo que pasa un quality gate.
- Detectar cuándo el modelo deja de funcionar y reentrenarlo.
- Documentar riesgos, controles y decisiones.

## Cómo empezar

1. Haz **fork** de este repositorio a tu cuenta.
2. Clona tu fork (reemplaza `TU-USUARIO` por tu usuario de GitHub) y crea el entorno:

```bash
git clone https://github.com/TU-USUARIO/mlops-churn-lab.git
cd mlops-churn-lab
git checkout -b lab/14.1

conda create -n churn-lab python=3.10 -y
conda activate churn-lab
pip install -r requirements.txt "setuptools<81"
```

3. Comprueba que todo corre:

```bash
pytest tests/ -q                      # 18 passed
python src/data_prep.py
python src/train.py --model all
python src/evaluate_and_register.py
uvicorn app.main:app --port 8000      # http://localhost:8000/docs
```

4. Sigue la guía: [`docs/11. mlops-churn-crisp-mlq-parte-1.md`](docs/11.%20mlops-churn-crisp-mlq-parte-1.md).

## Estructura

```
.
├── app/main.py                     # API FastAPI: /predict, /predict/batch, /health
├── src/
│   ├── data_prep.py                # limpieza, encoding, escalado y split
│   ├── train.py                    # 3 modelos con MLflow tracking
│   └── evaluate_and_register.py    # registro en MLflow Model Registry
├── tests/                          # tests de la API con el modelo mockeado
├── data/
│   └── telco_customer_churn_mlops.csv   # 300 clientes, 7 features, target churn
├── docs/                           # guía del laboratorio
├── .github/workflows/mlops.yml     # test → data-prep → train → register → smoke-test
├── Dockerfile
└── requirements.txt
```

## Dataset

| Columna | Tipo | Descripción |
|---|---|---|
| `customer_id` | texto | Identificador del cliente |
| `gender` | categoría | Male, Female |
| `senior_citizen` | 0/1 | Adulto mayor |
| `tenure_months` | entero | Meses como cliente |
| `contract_type` | categoría | Month-to-Month, One Year, Two Year |
| `internet_service` | categoría | Fiber, DSL, None |
| `monthly_charges` | decimal | Cargo mensual |
| `support_tickets_last_6m` | entero | Tickets de soporte en 6 meses |
| `churn` | Yes/No | El cliente abandonó (target) |

## Qué vas a construir

| Laboratorio | Fases CRISP-ML(Q) | Resultado |
|---|---|---|
| 14.1 | 1 a 4: negocio y datos, preparación, modelado, evaluación | Contrato de datos, Pipeline de scikit-learn, validación cruzada, umbral por costo, modelo `@champion` |
| 14.2 | 5 y 6: despliegue, monitoreo | API sin skew, imagen en Docker Hub, drift, reentrenamiento automático |

## Resultados del laboratorio 14.1 (retos)

### Reto 1 · Cuarto modelo candidato (`GradientBoostingClassifier`)

Validación cruzada estratificada de 5 folds sobre train (240 filas):

| Modelo | F1 CV (media ± desv.) | ROC-AUC CV |
|---|---|---|
| baseline | 0.6629 | 0.5000 |
| logistic | 0.6666 ± 0.0454 | 0.7315 |
| **random_forest** | **0.6900 ± 0.0426** | **0.7326** |
| xgboost | 0.6829 ± 0.0656 | 0.7100 |
| gradient_boosting | 0.6476 ± 0.0761 | 0.6858 |

`gradient_boosting` queda último entre los candidatos (incluso por debajo del baseline en F1) y **no cambia el modelo elegido**: sigue siendo `random_forest`.

### Reto 2 · Costos `cost_fn: 500`, `cost_fp: 50`

Umbral elegido: **0.05** (el mínimo de la grilla). El modelo marca a todos como churn (recall 1.0, precision 0.5), su costo es S/ 25.00 por cliente, igual que contactar a todos, y **falla `costo_vs_sin_modelo`** (25.0 no es < 25.0). Con un FN 10 veces más caro que un FP, el umbral de equilibrio es 50/550 ≈ 0.09: casi ninguna probabilidad cae por debajo, así que la política óptima es contactar a todos y el modelo no aporta ahorro.

### Reto 3 · `contract_type` en `quality_gate.slices`

| Contrato | n | Recall |
|---|---|---|
| Month-to-Month | 33 | 1.0000 |
| One Year | 16 | 0.5000 |
| Two Year | 11 | 0.6667 |

Brecha 0.50 > 0.25: **no cumple el gate**. Los contratos largos tienen pocos churners en test y el modelo aprendió que el contrato largo "protege". Propuesta: mantener el gate en atributos sensibles (género, edad) y monitorear `contract_type` aparte; mejorar con más datos de contratos largos, umbral por segmento o features de interacción, y exigir un `n` mínimo por grupo antes de evaluar la brecha.

### Reto 4 · Check de duplicados completos

`validate_data.py` agrega `no_full_duplicates` (filas idénticas ignorando `customer_id`) y `tests/test_data_contract.py::test_full_duplicate_rows_fail` lo prueba. El contrato pasa de 23 a 24 checks.

## Resultados del laboratorio 14.2 (retos)

### Reto 1 · Por qué `price_hike` se detecta sin etiquetas y `competitor` no

`python src/monitor.py --scenario <escenario>` (8 semanas, el drift empieza en la semana 5):

**price_hike** (sube la tarifa y se disparan los reclamos)

| ventana | psi_max | pred_psi | f1 | decisión |
|---|---|---|---|---|
| week_4 | 0.057 | 0.025 | 0.701 | MONITOREAR |
| week_5 | 0.584 | 0.066 | 0.716 | ALERTA |
| week_6 | 0.927 | 0.160 | 0.732 | ALERTA |
| week_7 | 2.418 | 0.182 | 0.723 | REENTRENAR |
| week_8 | 2.579 | 0.206 | 0.723 | REENTRENAR |

**competitor** (un competidor capta a los clientes de contrato largo)

| ventana | psi_max | pred_psi | f1 | decisión |
|---|---|---|---|---|
| week_4 | 0.057 | 0.025 | 0.701 | MONITOREAR |
| week_5 | 0.040 | 0.018 | 0.732 | MONITOREAR |
| week_6 | 0.019 | 0.020 | 0.507 | REENTRENAR |
| week_7 | 0.041 | 0.016 | 0.493 | REENTRENAR |
| week_8 | 0.055 | 0.028 | 0.384 | REENTRENAR |

- **`price_hike` es data drift**: cambian las features (`monthly_charges`, `support_tickets_last_6m`), es decir `P(X)`. El PSI se puede calcular con lo que llega a la API, sin saber quién se fue: la alerta sale en la semana 5 y REENTRENAR en la 7, con el F1 intacto. Con tráfico real (`simulate_traffic.py --scenario price_hike` + `monitor.py --from-log`) el PSI llega a 2.71 y la decisión es REENTRENAR sin una sola etiqueta.
- **`competitor` es concept drift**: los clientes siguen pareciéndose a los de entrenamiento (PSI de 0.02 a 0.05, todo "verde"), pero cambia la relación entre features y churn, `P(y|X)`: los de contrato largo ahora sí se van. Ni el PSI de las features ni el de las predicciones lo ven; solo el F1 contra la etiqueta real, que cae de 0.70 a 0.38.
- **Tiempo de reacción con etiquetas a 4 semanas**: con etiquetas inmediatas el sistema reacciona en la semana 6 (la primera ventana con F1 0.507). Si las etiquetas de la semana 6 llegan 4 semanas después, la caída se ve recién en la **semana 10**: **5 semanas después de que empezó el drift** (semana 5), 4 más que con etiquetas inmediatas. Durante ese tiempo el modelo sigue sin contactar a los clientes de contrato largo que se están yendo.

### Reto 2 · Ficha CRISP-ML(Q)

`docs/crisp-mlq.md` marca los controles de las fases 5 y 6 implementados y agrega los riesgos que esta solución todavía no cubre.

### Reto 3 · Alerta de PSI crítico

`monitoring/prometheus/alerts.yml` agrega `ChurnModelPsiCritical` (`model_psi_max > 0.25` durante 1 minuto). Con `monitor.py --scenario price_hike --push localhost:9091` el PSI máximo llega a 2.579 y la alerta pasa a FIRING. Con `competitor` no se dispara (PSI máximo 0.055): solo salta `ChurnModelRetrainNeeded` por la caída del F1.

### Reto 4 · Issue automático al reentrenar

`model-monitoring.yml` tiene `permissions: issues: write` y un paso que, si el monitor devuelve exit 42, abre un issue con la decisión, los motivos y el enlace a la corrida (`gh issue create`).

### Reto 5 · Rollback automático del circuit breaker

`scripts/deploy_ecs.sh staging "$REPOSITORY:no-existe"` registró la revisión 4 con una imagen inexistente. ECS intentó levantar la tarea varias veces (`CannotPullContainerError: ... churn-predictor:no-existe: not found`). Al llegar al umbral de fallos, el **circuit breaker** marcó el despliegue como `FAILED` ("tasks failed to start") y volvió solo a la revisión 3 ("rolling back to deployment ..."). La tarea anterior nunca se apagó, así que staging siguió respondiendo `/health` todo el tiempo.

El script termina con error por diseño: `aws ecs wait services-stable` agotó sus intentos (10 minutos) mientras el rollback seguía en curso (exit 255). Si el waiter hubiera alcanzado el estado estable, la verificación final habría encontrado la revisión 3 en ejecución en lugar de la 4 e impreso `El despliegue falló: ECS hizo rollback`, con exit 1. En los dos casos el job de despliegue falla y el pipeline no reporta un despliegue que no ocurrió.

### Retos 6 y 7 · Estado en S3 e `infra.yml`

No se ejecutaron en este laboratorio. Para el 6 bastaría con crear un bucket S3 versionado y una tabla DynamoDB de locks, descomentar el bloque `backend "s3"` de `infra/main.tf` y correr `terraform init -migrate-state`: los workspaces `staging` y `prod` pasan a `env:/staging/...` y `env:/prod/...` en el bucket. El 7 necesita ese estado remoto y otro usuario IAM con permisos de infraestructura.

### Reto 8 · Promover la misma imagen de staging a producción

En `main`, `build-and-publish` ya no reconstruye. Si el commit es un merge de `develop` (segundo padre `HEAD^2`) y su imagen existe en ECR, lee el manifiesto con `aws ecr batch-get-image` y lo vuelve a publicar con `aws ecr put-image` bajo los tags del commit de `main` y `latest`. Es el **mismo digest**, sin rebuild (el job tarda 12 s en vez de más de 1 minuto). `deploy-production` despliega ese tag y corre el test de skew con el modelo de la corrida de `develop` que construyó la imagen. Si el commit no viene de `develop` o la imagen no existe, el job construye como antes.

- **Qué gana el proyecto:** producción ejecuta exactamente el artefacto que se probó en staging (mismo digest, mismas dependencias, mismo `.pkl`). Desaparece el riesgo de que dos builds del mismo código den imágenes distintas (capas base actualizadas, versiones resueltas en otro momento, un modelo reentrenado con otra semilla). Además el release es más rápido y reproducible.
- **Qué cambia en el pipeline:** `main` deja de producir imágenes propias y depende de que la corrida de `develop` haya terminado con éxito. El reentrenamiento que dispara el monitor en `main` ya no cambia lo que se despliega: un modelo nuevo tiene que pasar primero por `develop` y staging.
