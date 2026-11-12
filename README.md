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
