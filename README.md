# Riesgo de abandono y valor del cliente bancario

Analítica de *churn* sobre una cartera de 10.000 clientes bancarios: limpieza, análisis exploratorio,
modelo predictivo de probabilidad de abandono, segmentación valor × riesgo con pérdida esperada y un
dashboard interactivo de dos páginas para operación de retención.

**Stack:** Python · pandas · scikit-learn · Plotly · Streamlit · Jupyter
**Dataset:** [Bank Customer Churn Records (Kaggle)](https://www.kaggle.com/datasets/adam036/customer-churn-records) — 10.000 filas, sintético y público

---

## 1. Objetivo

El banco necesitaba pasar de medir *cuántos* clientes se van a decidir **a quién debe atender primero y
cuánto dinero está en juego**. Los tres objetivos concretos:

1. **Cuantificar el riesgo individual.** Estimar `P(abandono)` por cliente con un modelo interpretable y
   bien calibrado, no solo una clasificación binaria.
2. **Priorizar por impacto económico, no por volumen.** Un cliente con 90% de probabilidad de irse y
   cuenta a cero no compete con uno de 60% y €150.000 de saldo. Objetivo: una métrica única de
   asignación de presupuesto → **pérdida esperada = `P(abandono) × Balance`**.
3. **Entregar la analítica en manos del equipo de retención.** Un scoring reproducible y un dashboard
   donde un gestor consulte un cliente concreto y obtenga el playbook aplicable a su perfil.

**Pregunta de negocio que responde el proyecto:** ¿qué 2.000 relaciones debo proteger este trimestre
para evitar la mayor parte de la fuga de depósitos?

---

## 2. Proceso

Cuatro etapas, cada una en su propio notebook y ejecutables en orden.

### 2.1 Limpieza y validación — `CustomerChurnRecordsClean.ipynb`

Auditoría del dataset original (10.000 × 18) y generación de variables derivadas.

| Comprobación | Resultado |
|---|---|
| Duplicados por `CustomerId` | 0 |
| Filas duplicadas completas | 0 |
| Valores nulos | 0 |
| Saldos negativos | 0 |
| Edades fuera de 18–100 | 0 |
| Salarios negativos | 0 |

**Hallazgo honesto:** el dataset ya venía íntegro, por lo que la limpieza **no eliminó ninguna fila**
(10.000 antes y después). Su valor real fue la validación de integridad y la ingeniería de dos
características para el análisis: `AgeGroup` (18-30 … 61+) y `BalanceCategory` (Cero / Bajo / Medio /
Alto, con cuartiles calculados solo sobre saldos > 0 para que el 36,2% de cuentas en cero no distorsione
los cortes).

### 2.2 Análisis exploratorio — `dataanalysis.ipynb`

Tasa de abandono global y desglose por demografía (geografía, género, edad), comportamiento
(actividad, productos, antigüedad) y valor (saldo, salario estimado).

### 2.3 Modelado y segmentación — `customer_analysis.ipynb`

**Modelo:** regresión logística (`class_weight="balanced"`, `max_iter=1000`), split 70/30 estratificado
con `random_state=42`, features estandarizadas y dummy coding (`drop_first`).

**Control de fuga de información.** `Complain` se excluye deliberadamente del entrenamiento: en este
dataset está casi alineado con la variable objetivo (99,5% de abandono con queja registrada), así que
incluirlo produciría un modelo inútil en producción. También se excluyen los identificadores
(`RowNumber`, `CustomerId`, `Surname`) y las columnas derivadas de la propia limpieza
(`AgeGroup`, `BalanceCategory`, `ValueRank`, `ValueTier`), que crearían redundancia con sus bases.

**Segmentación.** Terciles de `Balance` (valor) × terciles de `P(abandono)` (riesgo) → matriz 3×3 y
siete segmentos de acción con etiqueta de playbook.

### 2.4 Dashboard — `churn_dashboard.py`

Streamlit de dos páginas sobre el CSV de segmentos enriquecido con las columnas descriptivas:

- **Resumen ejecutivo:** KPIs de cartera, abandono por geografía / género / tipo de tarjeta, relación
  queja–satisfacción–abandono, matriz de calor valor × riesgo, pérdida esperada por segmento, curva de
  abandono por edad y cartera priorizada top 15.
- **Perfil y búsqueda:** localización por apellido o `CustomerId`, gauge de `P(abandono)`, radar contra
  su cohorte de edad (±4 años), ficha financiera y de relación, recomendaciones generadas por reglas y
  cola de clientes similares para el triaje semanal.

```bash
pip install -r requirements.txt
python customer_analysis.ipynb        # o ejecutar sus celdas en Jupyter
streamlit run churn_dashboard.py      # http://localhost:8501
```

---

## 3. Resultados

### 3.1 El modelo predice y está calibrado

| Métrica | Valor |
|---|---|
| AUC train | 0,825 |
| **AUC test** | **0,856** |
| Brier score test | 0,159 |
| Tasa de abandono real | 20,38% (2.038 de 10.000) |

Sin fuga de información y con solo variables transaccionales básicas, la capacidad discriminante es
alta y la calibración por deciles sigue la diagonal: las probabilidades pueden multiplicarse por el
saldo con sentido económico, que es justo lo que usa la pérdida esperada.

### 3.2 Concentración del riesgo: 27% de la cartera = 64% de la pérdida

Pérdida esperada total: **€361.929.265**.

| Segmento | Clientes | Churn real | Pérdida esperada | % pérdida | % abandonos |
|---|---:|---:|---:|---:|---:|
| Proteger (alto valor, alto riesgo) | 1.425 | 43,2% | €145.272.432 | **40,1%** | 30,2% |
| Retener selectivo (medio valor, alto riesgo) | 1.309 | 44,5% | €88.242.229 | 24,4% | 28,6% |
| Vigilar valor (alto valor, riesgo medio) | 1.216 | 12,8% | €64.568.996 | 17,8% | 7,7% |
| Estándar (riesgo medio) | 2.117 | 11,7% | €38.817.424 | 10,7% | 12,2% |
| Cultivar (alto valor, bajo riesgo) | 692 | 6,1% | €15.113.770 | 4,2% | 2,1% |
| Low-touch (bajo riesgo) | 2.642 | 2,9% | €9.914.415 | 2,7% | 3,7% |
| Gestionar coste (bajo valor, alto riesgo) | 599 | 52,9% | €0 | 0,0% | 15,6% |

- El **decil superior de riesgo** (1.000 clientes) concentra el **24,5%** de la pérdida esperada; el
  **top 20%** acumula el **44%**.
- El segmento *Gestionar coste* tiene el mayor churn (52,9%) y **€0 de pérdida esperada**: 599 clientes
  con cuenta a cero que no deben recibir ni un euro de presupuesto de retención. La métrica de pérdida
  esperada es la que hace visible esta trampa.

### 3.3 Cinco factores explican el abandono

**1 · Edad — punto de inflexión a los 40** (coeficiente estandarizado +0,747, el mayor del modelo)

| Banda | ≤30 | 30-40 | 40-45 | 45-50 | 50-55 | 55-60 | >60 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Churn | 7,5% | 12,1% | 26,8% | 46,4% | **58,4%** | 53,3% | 24,8% |

**2 · Inactividad con saldo alto** — la combinación más tóxica del dataset

| Perfil | Churn |
|---|---:|
| Inactivo con saldo (n=3.105) | **31,7%** |
| Inactivo sin saldo | 18,3% |
| Activo con saldo | 16,9% |
| Activo sin saldo | 9,6% |

Riesgo 3,3× mayor. El 48,5% de la cartera está inactiva y el saldo medio *crece* con el riesgo
(€97.080 en riesgo alto frente a €48.607 en riesgo bajo): **la cartera que se va es la que más dinero
tiene dentro**.

**3 · Alemania** — 32,4% de abandono (frente a 16,2% en Francia y 16,7% en España) sobre el mayor
depósito de valor: €300,4 M. Si el cliente alemán además está inactivo, el churn sube al **41,1%**.
Apunta a desajuste de producto y comisiones, no a precio aislado.

**4 · Número de productos — "exactamente dos"**

| Productos | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|
| Clientes | 5.084 | 4.590 | 266 | 60 |
| Churn | 27,7% | **7,6%** | 82,7% | 100% |

Pasar de uno a dos productos es el efecto de retención más potente y barato del dataset. A partir del
tercero, la relación se rompe.

**5 · Quejas** — 2.044 clientes con queja registrada y 99,5% de abandono. Excluida del modelo por fuga
de información, su valor es operativo: **la queja no se predice, se atiende**.

### 3.4 Dos hallazgos negativos que ahorraron presupuesto

- **La satisfacción no predice nada.** `Satisfaction Score` está distribuido de forma casi uniforme en
  escala 1–5 (1.932 / 2.014 / 2.042 / 2.008 / 2.004 clientes) y el churn es ~20% en todos los tramos
  (20,5% / 20,6% / 19,8%). La media es idéntica en los tres tiers de riesgo (3,04 / 3,04 / 2,97). El
  instrumento actual de CSAT **no discrimina** entre quien se va y quien se queda.
- **La lealtad acumulada tampoco protege.** `Tenure` medio de 5,0 años en los tres tiers. Tampoco
  aportan señal el tipo de tarjeta (churn entre 19,3% y 21,8% en las cuatro categorías), el score
  crediticio (−0,084), `HasCrCard` (−0,050) ni los puntos ganados.

### 3.5 Recomendaciones derivadas

| Prioridad | Audiencia | Acción |
|---|---|---|
| 1 | 1.425 clientes "Proteger" | Gestor dedicado, tope 60–80 cuentas, llamada en ≤15 días |
| 2 | 2.781 clientes de 40–55 | Trigger preventivo de re-evaluación vital a los 39–41 |
| 3 | 3.105 inactivos con saldo | Reactivación con objetivo de transacción recurrente a 90 días |
| 4 | 2.509 clientes alemanes | Task force de fit de producto y comisiones |
| 5 | 5.084 de un solo producto | Cross-sell del segundo producto |
| 6 | 326 con ≥3 productos | Congelar venta y abrir causa raíz |
| 7 | 599 "Gestionar coste" | Desinvertir: migrar a low-touch digital |
| 8 | Toda la cartera | Scoring mensual en CRM, presupuesto por pérdida esperada |

**Objetivos de 12 meses:** churn global ≤17% · retención en "Proteger" ≥70% · ≥€40 M de pérdida
esperada mitigada · churn Alemania ≤25% · ≥85% de quejas cerradas en <48h.

---

## 4. Estructura del repositorio

```
.
├── Customer-Churn-Records.csv            # dataset original (10.000 × 18)
├── Customer-Churn-Records-clean.csv      # validado + AgeGroup y BalanceCategory
├── Customer-Risk-Value-Segments.csv      # salida del modelado: scoring y segmentos
├── CustomerChurnRecordsClean.ipynb       # etapa 1 · limpieza y validación
├── dataanalysis.ipynb                    # etapa 2 · análisis exploratorio
├── customer_analysis.ipynb               # etapa 3 · modelo, segmentación, pérdida esperada
├── churn_dashboard.py                    # etapa 4 · dashboard Streamlit
├── requirements.txt
└── README.md
```

`Customer-Risk-Value-Segments.csv` es una salida regenerable: se produce ejecutando
`customer_analysis.ipynb`. Se incluye para que el dashboard funcione sin pasos previos.

## 5. Limitaciones conocidas

- **Balance como proxy de valor.** La pérdida esperada trata el saldo como valor económico. Con margen
  real por cliente (comisiones netas del coste de fondos) la priorización mejoraría.
- **Monotonicidad forzada en edad.** La regresión logística no puede capturar la caída del churn tras
  los 60 años; un gradient boosting calibrado probablemente suba la AUC y modele mejor `NumOfProducts`.
- **Sesgo de supervivencia.** El descenso del churn a >60 años puede reflejar una cartera que ya
  envejeció dentro del banco, no un comportamiento actual.
- **Sin validación temporal.** El split es aleatorio estratificado; en producción conviene un corte por
  fecha para simular el uso real del modelo.
- **Cada campaña debe medirse con grupo de control** (15–20% aleatorio). Sin control no se distingue
  retención efectiva de clientes que se habrían quedado igual.


