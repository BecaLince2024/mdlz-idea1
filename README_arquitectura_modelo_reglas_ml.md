
# Arquitectura del **Modelo a Prueba de Balas – Reglas + ML**

Este documento describe la arquitectura completa del pipeline híbrido que combina **reglas de negocio automáticas** con **modelos de Machine Learning de respaldo**, basada en tu script de Python.

---

## 1. Capas de la Arquitectura

### 1.1 Capa de Entrada de Datos

**Responsables principales:**
- `DATA_PATH = "final_mapping.xlsx"` → datos históricos etiquetados (training).
- `NEW_DATA_PATH = "new_data.csv"` → datos nuevos sin consolidar (predicción).

**Funciones:**
- `load_data(path, is_csv=False)`  
  - Carga Excel usando `engine='calamine'` si está disponible; si no, el engine por defecto.
  - Carga CSV usando `engine='pyarrow'` si está disponible; si no, el engine por defecto.

---

### 1.2 Capa de Preprocesamiento

**Función clave:**
- `preprocess(df: pd.DataFrame) -> pd.DataFrame`

**Qué hace:**
1. Normaliza nombres de columnas a **UPPERCASE** (`df.columns.str.upper().str.strip()`).
2. Para cada columna en `FEATURE_COLS = ["MANUFACTURER_TYPE", "COUNTRY", "CATEGORY", "SEGMENT", "BRAND"]`:
   - Castea a `str`.
   - Pasa a **UPPERCASE**.
   - Elimina espacios al inicio y al final.

**Objetivo:**  
Garantizar que las reglas y el ML trabajen sobre un **espacio de valores limpio y consistente**, sin depender de diferencias de mayúsculas/minúsculas o espacios.

---

### 1.3 Capa de Reglas de Negocio – `BusinessRulesExtractor`

Clase: `BusinessRulesExtractor`

**Responsabilidad:**  
Extraer reglas de negocio directamente de los datos históricos para maximizar la precisión en los casos “claros” y delegar al ML solo la incertidumbre.

#### 1.3.1 Tipos de reglas extraídas

1. **Reglas de Segmento**
   - Forma: `SEGMENT -> CONSOLIDATEDSEGMENT`
   - Lógica:
     - Para cada `SEGMENT`, se calcula la distribución de `TARGET_SEGMENT` (CONSOLIDATEDSEGMENT).
     - Se toma la clase más frecuente (`most_common`).
     - Se calcula la confianza: `confidence = most_common_count / total`.
     - Si `confidence ≥ 0.90`, se guarda la regla.
   - Almacenamiento:
     - `self.segment_rules[segment] = most_common`

2. **Reglas de Marca Específicas**
   - Forma: `(MANUFACTURER_TYPE, BRAND) -> CONSOLIDATEDBRAND`
   - Lógica:
     - Se agrupa por `["MANUFACTURER_TYPE", "BRAND"]`.
     - Se calcula la distribución de `TARGET_BRAND` (CONSOLIDATEDBRAND).
     - Se toma la clase más frecuente y su confianza.
     - Si `confidence ≥ 0.90`, se guarda la regla.
   - Almacenamiento:
     - `self.brand_rules[(manu, brand)] = most_common`

3. **Reglas Default por Fabricante**
   - Forma: `MANUFACTURER_TYPE -> CONSOLIDATEDBRAND (default)`
   - Lógica:
     - Para cada `MANUFACTURER_TYPE`, se mira la distribución de `TARGET_BRAND`.
     - Se toma la clase más frecuente.
     - Si `confidence ≥ 0.80`, se guarda como regla default del fabricante.
   - Almacenamiento:
     - `self.manufacturer_rules[manu] = most_common`

#### 1.3.2 Aplicación de reglas

- `apply_segment_rule(segment)`  
  - Devuelve `(predicción, método)`:
    - Si hay regla: `(consolidated_segment, "REGLA")`
    - Si no hay: `(None, "ML")`

- `apply_brand_rule(manufacturer, brand)`  
  - Devuelve `(predicción, método)`:
    - Si existe regla específica `(manu, brand)`: `(consolidated_brand, "REGLA")`
    - Si no, pero hay regla default por `MANUFACTURER_TYPE`: `(consolidated_brand, "REGLA_DEFAULT")`
    - Si no hay nada: `(None, "ML")`

#### 1.3.3 Persistencia de reglas

- `save_rules(RULES_PATH)` genera un archivo de texto (`reglas_extraidas.txt`) con:
  - Lista de reglas de segmento.
  - Lista de reglas default por fabricante.
  - Conteos totales de reglas extraídas.

---

### 1.4 Capa de ML de Respaldo – `BackupMLModel`

Clase: `BackupMLModel`

**Rol:**  
Modelo de Machine Learning que entra en juego **solo cuando las reglas no se pueden aplicar**.

#### 1.4.1 Instancias

- `self.ml_segment` → predice `CONSOLIDATEDSEGMENT` donde no hay regla de segmento.
- `self.ml_brand` → predice `CONSOLIDATEDBRAND` donde no hay regla de marca.

#### 1.4.2 Codificación

- `self.label_encoder` para el target.
- `self.feature_encoders[col]` (un `LabelEncoder` por feature categórica).

#### 1.4.3 Motor de ML (jerarquía)

En `fit`:

1. Si `LIGHTGBM_AVAILABLE`:
   - Se usa `lgb.LGBMClassifier` con:
     - `objective = "multiclass"` o `"binary"` según el número de clases.
     - `class_weight = "balanced"`.
     - `num_leaves = 64`, `max_depth = 8`, `n_estimators = 500`.
     - `device = "gpu"` si hay GPU.
2. En caso contrario, si `CATBOOST_AVAILABLE`:
   - Se usa `CatBoostClassifier` con:
     - `loss_function = "MultiClass"`.
     - `auto_class_weights = "Balanced"`.
     - `task_type = "GPU"` o `"CPU"` según `use_gpu`.
3. Si no hay ninguno de los dos:
   - Se usa `RandomForestClassifier` de sklearn con:
     - `n_estimators = 200`, `max_depth = 10`, `class_weight = "balanced"`.

> Importante: **solo se usa un modelo por target**, no compiten entre sí.

#### 1.4.4 Predicción

- Para LightGBM / RandomForest:
  - Se codifican las features con `feature_encoders`.
  - Se llama a `model.predict` y se aplica `label_encoder.inverse_transform`.
- Para CatBoost:
  - Se construye un `Pool` con columnas categóricas.
  - `model.predict(pool)` produce las clases directamente.

---

### 1.5 Capa Híbrida – `HybridPredictor`

Clase: `HybridPredictor`

**Responsabilidad global:**  
Orquestar las reglas + ML sobre el training y sobre nuevos datos.

#### 1.5.1 Entrenamiento (`fit(df)`)

1. Creación del extractor de reglas:
   - `self.rules_extractor = BusinessRulesExtractor(df)`
   - `self.rules_extractor.extract_all_rules()`
   - `self.rules_extractor.save_rules(RULES_PATH)`

2. Entrenamiento de `ml_segment`:
   - Se listan `segments_sin_regla` = segmentos sin entrada en `segment_rules`.
   - Se filtra `df` para quedarse solo con esos segmentos.
   - Se entrena `BackupMLModel` **solo con las filas de segmentos sin regla**.

3. Entrenamiento de `ml_brand`:
   - Para cada fila se evalúa `has_brand_rule(row)`:
     - Usa `apply_brand_rule` y verifica si devuelve predicción.
   - Se entrena `BackupMLModel` **solo con filas sin regla de marca**.

4. Evaluación interna (`_evaluate(df)`):
   - Se predice `CONSOLIDATEDSEGMENT` y `CONSOLIDATEDBRAND` para **todo el df_train** usando:
     - Reglas → ML → fallback (misma lógica que para datos nuevos).
   - Se calculan:
     - `accuracy_score` y `f1_score (macro)` para ambos targets.
     - Conteo de cuántas predicciones usaron:
       - Reglas (`"REGLA"`, `"REGLA_DEFAULT"`).
       - ML (`"ML"`).

   - Las métricas se guardan en `self.metrics`:
     - `segment_accuracy`, `segment_f1`, `brand_accuracy`, `brand_f1`.

---

### 1.6 Lógica de Predicción Fila a Fila

#### 1.6.1 Predicción de Segmento – `predict_segment(df)`

Para cada fila:

1. Se intenta `apply_segment_rule(SEGMENT)`:
   - Si devuelve una predicción:
     - `SEGMENT_pred = regla`
     - `SEGMENT_method = "REGLA"`
2. Si NO hay regla, pero `ml_segment` está entrenado:
   - Se construye un mini DataFrame `row_df` con `FEATURE_COLS`.
   - `SEGMENT_pred = ml_segment.predict(row_df)[0]`
   - `SEGMENT_method = "ML"`
3. Si NO hay regla ni modelo:
   - `SEGMENT_pred = SEGMENT.title()`
   - `SEGMENT_method = "FALLBACK"`

#### 1.6.2 Predicción de Marca – `predict_brand(df)`

Para cada fila:

1. Se intenta `apply_brand_rule(MANUFACTURER_TYPE, BRAND)`:
   - Si devuelve predicción:
     - `BRAND_pred = regla`
     - `BRAND_method = "REGLA"` o `"REGLA_DEFAULT"`
2. Si NO hay regla, pero `ml_brand` está entrenado:
   - `BRAND_pred = ml_brand.predict(row_df)[0]`
   - `BRAND_method = "ML"`
3. Si NO hay regla ni modelo:
   - Si `MANUFACTURER_TYPE == "COMPETITOR"`:
     - `BRAND_pred = "Competitor"`
     - `BRAND_method = "FALLBACK_COMPETITOR"`
   - En caso contrario:
     - `BRAND_pred = BRAND.title()`
     - `BRAND_method = "FALLBACK_BRAND"`

#### 1.6.3 Cálculo de Confianza

En `predict(df_new)` se genera la columna `CONFIANZA` a partir de los métodos de segmento y marca:

- Si `SEGMENT_method == "REGLA"` **y** `"REGLA"` en `BRAND_method` → **🟢 ALTA**
- Si `SEGMENT_method == "REGLA"` **o** `"REGLA"` en `BRAND_method` → **🟡 MEDIA**
- Si `"ML"` en alguno de los métodos → **🟠 BAJA**
- Si solo hay fallbacks → **🔴 MUY BAJA**

---

## 2. Diagrama del Pipeline Principal (Mermaid)

```mermaid
flowchart TD
    A[Inicio main()] --> B[Detectar hardware<br/>detect_gpu()]
    B --> C[Cargar df_train<br/>final_mapping.xlsx<br/>load_data + preprocess]
    C --> D{Existe new_data.csv?}
    D -->|Sí| E[Cargar df_new<br/>load_data + preprocess]
    D -->|No| F[df_new = None]

    C --> G[Crear HybridPredictor(use_gpu)]
    G --> H[fit(df_train)<br/>Entrenar sistema híbrido]

    H --> I[Evaluar en df_train<br/>_evaluate(df_train)<br/>métricas + % reglas vs ML]

    I --> J{df_new no es None?}
    J -->|Sí| K[predict(df_new)<br/>df_result]
    J -->|No| N[Omitir predicción de nuevos datos]

    K --> L[Imprimir predicciones detalladas<br/>por fila]
    L --> M[Guardar df_result<br/>new_data_with_predictions.xlsx]

    M --> O[Imprimir resumen final<br/>accuracy y F1 de ambos targets]
    N --> O

    O --> P[Guardar predictor.pkl]
    P --> Q[Fin del proceso]
```

---

## 3. Diagrama Interno de Entrenamiento Híbrido

```mermaid
flowchart TD
    A[HybridPredictor.fit(df_train)] --> B[Crear BusinessRulesExtractor(df_train)]
    B --> C[extract_all_rules()<br/>SEGMENT, BRAND, MANUFACTURER_TYPE]
    C --> D[save_rules(reglas_extraidas.txt)]

    D --> E[Entrenar ML para CONSOLIDATEDSEGMENT]
    E --> F[Encontrar segments_sin_regla<br/>SEGMENT sin regla]
    F --> G{¿Hay filas sin regla?}
    G -->|Sí| H[Construir X_ml,y_ml<br/>solo filas sin regla]
    H --> I[Crear BackupMLModel(TARGET_SEGMENT)]
    I --> J[BackupMLModel.fit(X_ml,y_ml)]
    G -->|No| K[ml_segment = None<br/>No se necesita ML]

    J --> L[Entrenar ML para CONSOLIDATEDBRAND]
    K --> L

    L --> M[Identificar filas sin regla de BRAND<br/>apply_brand_rule() es None]
    M --> N{¿Hay filas sin regla?}
    N -->|Sí| O[Construir X_ml,y_ml<br/>solo filas sin regla de marca]
    O --> P[Crear BackupMLModel(TARGET_BRAND)]
    P --> Q[BackupMLModel.fit(X_ml,y_ml)]
    N -->|No| R[ml_brand = None<br/>No se necesita ML]

    Q --> S[_evaluate(df_train)<br/>Predict en todo df_train<br/>calcular accuracy y F1]
    R --> S
    S --> T[Guardar métricas en self.metrics]
```

---

## 4. Diagrama Fila a Fila: Reglas + ML + Fallback

```mermaid
flowchart TD
    A[Fila de datos] --> B[Predicción de SEGMENT]
    B --> C[apply_segment_rule(SEGMENT)]
    C --> D{¿Existe regla para SEGMENT?}
    D -->|Sí| E[SEGMENT_pred = regla<br/>SEGMENT_method = "REGLA"]
    D -->|No| F{¿ml_segment entrenado?}
    F -->|Sí| G[ML predice CONSOLIDATEDSEGMENT<br/>SEGMENT_method = "ML"]
    F -->|No| H[SEGMENT_pred = SEGMENT.title()<br/>SEGMENT_method = "FALLBACK"]

    A --> I[Predicción de BRAND]
    I --> J[apply_brand_rule(MANUFACTURER_TYPE, BRAND)]
    J --> K{¿Regla específica o default?}
    K -->|Sí| L[BRAND_pred = regla<br/>BRAND_method = "REGLA" o "REGLA_DEFAULT"]
    K -->|No| M{¿ml_brand entrenado?}
    M -->|Sí| N[ML predice CONSOLIDATEDBRAND<br/>BRAND_method = "ML"]
    M -->|No| O{MANUFACTURER_TYPE == "COMPETITOR"?}
    O -->|Sí| P[BRAND_pred = "Competitor"<br/>BRAND_method = "FALLBACK_COMPETITOR"]
    O -->|No| Q[BRAND_pred = BRAND.title()<br/>BRAND_method = "FALLBACK_BRAND"]

    E & G & H & L & N & P & Q --> R[Calcular CONFIANZA<br/>🟢 ALTA / 🟡 MEDIA / 🟠 BAJA / 🔴 MUY BAJA]
```

---

## 5. Mensaje Clave para Stakeholders

- **Reglas primero:** el sistema explota el conocimiento “duro” de la base histórica para obtener predicciones ultra confiables donde los patrones son claros.
- **ML como red de seguridad:** los modelos solo se activan cuando las reglas no pueden tomar una decisión robusta.
- **Fallback controlado:** incluso sin reglas ni ML, el sistema siempre produce un resultado trazable.
- **Interpretabilidad:** cada predicción viene acompañada del **método** (`REGLA`, `REGLA_DEFAULT`, `ML`, `FALLBACK_*`) y un nivel de **confianza** fácilmente explicable al negocio.

Este documento se puede usar como:
- README técnico del repositorio.
- Anexo explicativo para stakeholders no técnicos.
- Base para construir slides de presentación.
