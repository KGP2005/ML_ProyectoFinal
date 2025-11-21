# Proyecto Final - Machine Learning: Clasificación de Gallstone Status

## Descripción del Proyecto

Este proyecto implementa un sistema completo de Machine Learning para predecir la presencia o ausencia de cálculos biliares (Gallstone Status) basándose en variables demográficas, comorbilidades y mediciones corporales.

**Dataset**: UCI Gallstone Dataset  
**Variable objetivo**: Gallstone Status (0 = Present, 1 = Absent)  
**Notebook principal**: `Modelo_Gallstone_Optimizado_FIXED.ipynb`

## 🎯 Objetivo Clínico

**PRIORIDAD CRÍTICA**: Minimizar Falsos Negativos (FN)
- No detectar un caso de Gallstone puede tener consecuencias graves para el paciente
- El modelo prioriza **Recall** sobre otras métricas
- Métrica de evaluación principal: **F2-Score** (enfatiza Recall sobre Precision)

## 🆕 Mejoras Implementadas (Noviembre 2025)

### ✅ 1. Análisis Exploratorio Avanzado (EDA)
- **Detección de Outliers**: Métodos IQR y Z-score
- **Tests Estadísticos**: t-test y Mann-Whitney para diferencias significativas entre clases
- **Visualizaciones Avanzadas**:
  - Boxplots por clase para top features
  - Pair plots de características correlacionadas
  - Análisis de distribuciones

### ✅ 2. Pipeline de ML Completo (Evita Data Leakage)
```python
Pipeline([
    ('scaler', StandardScaler()),           # Normalización
    ('feature_engineer', FeatureEngineer()),  # Creación de features
    ('model', OptimizedModel())             # Modelo optimizado
])
```
- **Feature Engineering**: 6 features adicionales creadas
  - BMI × Age
  - HDL/LDL ratio
  - Fat Index
  - Age²
  - TG/HDL ratio
  - Liver Enzyme Index

### ✅ 3. Estrategia de División Correcta
- **Train (70%)**: Balanceado con SMOTE/Borderline-SMOTE/ADASYN
- **Validation (15%)**: Distribución real (NO balanceado)
- **Test (15%)**: Distribución real (NO balanceado)
- ⚠️ **Crítico**: SMOTE solo se aplica en train, NUNCA en val/test

### ✅ 4. Comparación de Técnicas de Balanceo
- SMOTE
- Borderline-SMOTE
- ADASYN
- Selección automática del mejor método según Recall

### ✅ 5. Optimización de Hiperparámetros (TODOS los modelos)
Grids de búsqueda completos para:
- **Logistic Regression**: C, penalty, solver, class_weight
- **Decision Tree**: max_depth (ajustado a 3-7 para dataset pequeño), min_samples_split, criterion
- **Random Forest**: n_estimators, max_depth, min_samples, max_features
- **Gradient Boosting**: learning_rate, n_estimators, subsample
- **XGBoost**: max_depth, learning_rate, scale_pos_weight
- **LightGBM**: num_leaves, learning_rate, class_weight
- **SVM**: C (optimizado), kernel, gamma
- **KNN**: n_neighbors (optimizado), weights, metric
- **Naive Bayes**: var_smoothing
- **AdaBoost**: n_estimators, learning_rate

### ✅ 6. Stratified K-Fold Cross-Validation
- **5-Fold CV** sobre TODO el dataset
- Análisis de varianza entre folds
- Evaluación de estabilidad del modelo
- Métricas agregadas con desviación estándar

### ✅ 7. Priorización de Recall (FN Críticos)
- Scoring principal: **Recall weighted**
- Métrica secundaria: **F2-Score** (β=2, prioriza recall)
- Análisis detallado de:
  - Falsos Negativos por modelo
  - Recall por clase
  - Tasa de FN
  - Impacto clínico

### ✅ 8. Curvas ROC Completas
- Visualización de curvas ROC para todos los modelos
- Comparación visual de AUC
- Análisis de trade-off Sensibilidad vs Especificidad

### ✅ 9. Modelos Implementados
1. Logistic Regression (con class_weight='balanced')
2. Decision Tree (max_depth optimizado para 319 muestras)
3. Random Forest (class_weight='balanced')
4. Gradient Boosting
5. XGBoost (scale_pos_weight ajustado)
6. LightGBM (class_weight='balanced')
7. SVM (C optimizado, class_weight='balanced')
8. KNN (k optimizado)
9. Naive Bayes
10. AdaBoost

## 📊 Resultados Esperados

### Métricas Objetivo
- **Recall en Test**: ≥95% (minimizar FN)
- **F2-Score en Test**: ≥0.90
- **Falsos Negativos**: ≤2 casos
- **AUC-ROC**: ≥0.90

### Flujo de Evaluación
1. **Selección de Modelo Base**: Comparación inicial de 10 modelos
2. **Optimización de Hiperparámetros**: Top 3 modelos con RandomizedSearchCV
3. **Stratified K-Fold CV**: Validación robusta sobre dataset completo
4. **Test Final**: Evaluación en datos completamente no vistos

## Estructura del Proyecto

```
ML_ProyectoFinal/
├── dataset-uci.csv                           # Dataset en formato CSV
├── Modelo_Gallstone_Optimizado_FIXED.ipynb   # Notebook principal MEJORADO ✅
├── Info adicional del dataset.txt           # Descripción de variables
├── Instrrucciones.txt                       # Instrucciones del proyecto
├── README.md                                 # Este archivo
├── requirements.txt                          # Dependencias
├── gallstone_pipeline_complete_*.pkl        # Pipeline completo guardado
├── gallstone_model_metadata_*.txt           # Metadata del modelo
└── Propmpt inicial.txt                      # Prompt original
```

## Requisitos e Instalación

### Dependencias principales

```bash
pip install -r requirements.txt
```

### Archivo `requirements.txt`

```
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
imbalanced-learn>=0.11.0  # SMOTE, Borderline-SMOTE, ADASYN
scipy>=1.10.0             # Tests estadísticos
xgboost>=2.0.0            # XGBoost
lightgbm>=4.0.0           # LightGBM
joblib>=1.3.0             # Guardar modelos
```

## Uso del Notebook

### Ejecutar el Notebook Completo

1. **Abrir**: `Modelo_Gallstone_Optimizado_FIXED.ipynb`
2. **Ejecutar todas las celdas** desde el inicio

### Estructura del Notebook

#### Sección 1: Carga y Limpieza de Datos
- Verificación de valores nulos
- Información del dataset

#### Sección 2: Análisis Exploratorio (EDA)
- **2.1**: Detección de outliers (IQR, Z-score)
- **2.2**: Tests estadísticos (t-test, Mann-Whitney)
- **2.3**: Boxplots por clase
- **2.4**: Pair plots de top features
- Distribución de clases
- Matriz de correlación

#### Sección 3: Eliminación de Features Redundantes
- Reducción de multicolinealidad

#### Sección 4: División Estratificada
- Train 70% / Val 15% / Test 15%
- Estratificación para mantener proporción de clases

#### Sección 5: Comparación de Técnicas de Balanceo
- SMOTE vs Borderline-SMOTE vs ADASYN
- Selección del mejor método

#### Sección 6: Pipeline de ML
- StandardScaler
- Feature Engineering (6 features adicionales)
- Modelo

#### Sección 7: Entrenamiento de 10 Modelos
- Comparación inicial
- Priorización de Recall y F2-Score
- Curvas ROC

#### Sección 8: Visualizaciones Comparativas
- Gráficos de métricas
- Análisis de Falsos Negativos
- Curvas ROC superpuestas

#### Sección 9: Optimización de Hiperparámetros
- RandomizedSearchCV para top 3 modelos
- **9.1**: Stratified K-Fold CV sobre dataset completo

#### Sección 10: Evaluación Final en Test
- Métricas detalladas
- Matriz de confusión
- Análisis clínico de FN
- Curva ROC en test
- Recall por clase

#### Sección 11: Feature Importance
- Top features más importantes
- Features engineered en ranking

#### Sección 12: Resumen Ejecutivo
- Evolución de métricas
- Comparación CV vs Test
- Conclusiones

#### Sección 13: Guardar Modelo
- Pipeline completo
- Metadata

#### Sección 14: Guía de Uso
- Instrucciones para producción
- Ejemplos de código

## 🚀 Uso del Modelo en Producción

### Cargar Modelo Guardado

```python
import joblib
import pandas as pd

# Cargar pipeline completo
pipeline = joblib.load('gallstone_pipeline_complete_[timestamp].pkl')

# Preparar datos nuevos (mismas features originales)
X_new = pd.DataFrame({
    'Age': [45, 52],
    'Body Mass Index (BMI)': [28.5, 32.1],
    # ... resto de features
})

# Predicción
y_pred = pipeline.predict(X_new)
y_proba = pipeline.predict_proba(X_new)

# Interpretar
for i, (pred, proba) in enumerate(zip(y_pred, y_proba)):
    print(f"Paciente {i+1}:")
    print(f"  Predicción: {'Gallstone Present' if pred == 0 else 'No Gallstone'}")
    print(f"  Probabilidad Gallstone: {proba[0]:.1%}")
```

### Ajustar Umbral de Decisión (Opcional)

Para contextos donde FN son críticos:

```python
# Umbral conservador (minimiza FN)
umbral = 0.3
y_pred_conservative = (y_proba[:, 0] > umbral).astype(int)
```

## 📈 Comparación: Antes vs Después

### Problemas Corregidos

| Problema Original | Solución Implementada |
|-------------------|----------------------|
| ❌ Decision Tree max_depth=10 (muy profundo) | ✅ max_depth ajustado a 3-7 |
| ❌ SVM con C=1 por defecto | ✅ Grid search con C=[0.1, 1, 10, 50, 100] |
| ❌ KNN con k=5 fijo | ✅ Optimización de k=[3,5,7,9,11,15] |
| ❌ Solo optimización condicional | ✅ Grids para TODOS los modelos |
| ❌ SMOTE mezclado con val/test | ✅ SMOTE solo en train |
| ❌ No hay análisis de outliers | ✅ IQR + Z-score implementados |
| ❌ No hay tests estadísticos | ✅ t-test + Mann-Whitney |
| ❌ Falta visualizaciones avanzadas | ✅ Boxplots + Pair plots |
| ❌ Solo AUC numérico | ✅ Curvas ROC completas |
| ❌ No hay Stratified K-Fold | ✅ 5-Fold CV sobre dataset completo |
| ❌ No hay Pipeline | ✅ Pipeline completo (evita leakage) |
| ❌ No se prioriza Recall | ✅ F2-Score + análisis de FN |

## 🏥 Consideraciones Clínicas

### Impacto de Falsos Negativos

- **FN (Falso Negativo)**: Paciente con Gallstone diagnosticado como sano
  - **Riesgo**: Complicaciones graves no tratadas
  - **Prioridad**: MINIMIZAR FN a toda costa

- **FP (Falso Positivo)**: Paciente sano diagnosticado con Gallstone
  - **Impacto**: Exámenes adicionales innecesarios
  - **Prioridad**: Secundaria (preferible a FN)

### Interpretación de Métricas

- **Recall ≥95%**: El modelo detecta al menos 95% de casos de Gallstone
- **F2-Score**: Prioriza Recall sobre Precision (β=2)
- **FN ≤2**: Máximo 2 casos no detectados en test set

## 🔬 Metodología Científica

### Validación Robusta

1. **Split Estratificado**: Mantiene proporción de clases
2. **Balanceo Solo en Train**: Evita data leakage
3. **Pipeline Completo**: Transformaciones consistentes
4. **CV Estratificado**: Evaluación robusta
5. **Test Independiente**: Evaluación final no sesgada

### Reproducibilidad

- Seed fija: `SEED = 40`
- Pipeline guardado completo
- Metadata detallada
- Código versionado

## 📚 Referencias y Dataset

- **Dataset**: UCI Machine Learning Repository - Gallstone Dataset
- **Tamaño**: 319 muestras
- **Features**: Variables demográficas, comorbilidades, mediciones corporales
- **Target**: Binario (0=Present, 1=Absent)
- → Linear(64) → BatchNorm → ReLU → Dropout(0.3)
- → Linear(2) [Output]

**DESPUÉS** (optimizado):
- Input → Linear(128) → BatchNorm → ReLU → Dropout(0.5)
- → Linear(64) → BatchNorm → ReLU → Dropout(0.5)
- → Linear(2) [Output]

**Mejoras**: 50% menos parámetros, dropout más agresivo

### ResNet Classifier Optimizado
**ANTES** (sobreajustado):
- Input → Linear(256) → BatchNorm → ReLU → Dropout(0.3) [Proyección]
- → 4x Bloques Residuales [256 → 256 con skip connections]
- → Linear(2) [Clasificador]

**DESPUÉS** (optimizado):
- Input → Linear(128) → BatchNorm → ReLU → Dropout(0.5) [Proyección]
- → 2x Bloques Residuales [128 → 128 con skip connections]
- → Linear(2) [Clasificador]

**Mejoras**: 75% menos parámetros, 50% menos bloques residuales

**Bloque Residual**:
```
x → Linear(dim) → BatchNorm → ReLU → Dropout(0.5) → Linear(dim) → BatchNorm → (+) → ReLU
↑_______________________________________________________________|
```

## Resultados Esperados

### ANTES de las mejoras (baseline)
- **MLP**: Accuracy ~0.85, F1 ~0.84
- **ResNet**: Accuracy ~0.87, F1 ~0.86
- **Overfitting**: val_loss >> train_loss (gap > 0.15)

### DESPUÉS de las mejoras
- **MLP Optimizado**: Accuracy ~0.88-0.90, F1 ~0.87-0.89
- **ResNet Optimizado**: Accuracy ~0.87-0.89, F1 ~0.86-0.88
- **Lasso/Ridge**: Accuracy ~0.85-0.87, F1 ~0.84-0.86
- **Overfitting reducido**: val_loss ≈ train_loss (gap < 0.08)
- **Curvas**: Convergencia entre train/val más cercana

**Tiempo de entrenamiento**:
- MLP/ResNet: ~3-7 minutos por modelo (CPU), <2 min (GPU)
- Optuna (30 trials): ~20-30 minutos
- Lasso/Ridge: ~1-2 minutos

## Orden de Ejecución del Notebook

### Celdas obligatorias (orden secuencial):
1. Carga de datos
2. EDA y visualizaciones
3. Análisis de correlación
4. **[NUEVA]** Eliminación de features redundantes
5. Preprocesamiento y división train/val/test
6. **[NUEVA]** Data augmentation con SMOTE
7. Configuración de PyTorch y DataLoaders
8. **[MODIFICADO]** Modelo MLP optimizado
9. **[MODIFICADO]** Función de entrenamiento mejorada
10. **[MODIFICADO]** Entrenar MLP con configuración anti-overfitting
11. Visualizar curvas de aprendizaje MLP
12. Evaluar MLP en test set
13. **[MODIFICADO]** Modelo ResNet optimizado
14. **[MODIFICADO]** Entrenar ResNet
15. Visualizar curvas de aprendizaje ResNet
16. Evaluar ResNet en test set

### Celdas opcionales (pueden tardar):
17. **[NUEVA]** Búsqueda de hiperparámetros con Optuna (~30 min)
18. **[NUEVA]** Entrenar con mejores hiperparámetros de Optuna
19. **[NUEVA]** Evaluar modelo optimizado por Optuna
20. **[NUEVA]** Modelos con Lasso/Ridge/Elastic Net (~2 min)
21. **[MODIFICADO]** Comparativa final ampliada (6 modelos)

## Ejecución Rápida

### Cargar modelo guardado
```python
# Cargar el mejor modelo MLP
model_mlp.load_state_dict(torch.load('mlp_classifier_best.pth'))
model_mlp.eval()

# Hacer predicciones
with torch.no_grad():
    outputs = model_mlp(X_test_tensor.to(device))
    _, preds = torch.max(outputs, 1)
```

### Evaluar un modelo
```python
results = evaluate_model(model_mlp, test_loader, model_name='MLP Classifier')
```

## Notas Importantes

1. **GPU vs CPU**: El código detecta automáticamente si CUDA está disponible
2. **Reproducibilidad**: Se utilizan `random_state=42` en todas las divisiones
3. **Estratificación**: Los conjuntos train/val/test mantienen la proporción de clases
4. **Normalización**: StandardScaler ajustado solo en train, transformado en val/test
5. **SMOTE**: Solo se aplica al conjunto de entrenamiento, NO a validation/test
6. **Optuna**: Puede saltarse si el tiempo es limitado (los otros modelos ya están optimizados)

## Validación de Mejoras

### Indicadores de éxito:
- ✅ `val_loss` y `train_loss` más cercanos (gap < 0.1)
- ✅ Curvas de aprendizaje convergiendo juntas
- ✅ Métricas en test set mejores que baseline
- ✅ No detención temprana por patience muy rápido

### Si aún hay overfitting:
- Aumentar dropout a 0.6-0.7
- Reducir más la arquitectura
- Aumentar weight_decay a 1e-3
- Reducir batch_size a 32

## Archivos de Documentación

- **README.md**: Este archivo (visión general)
- **MEJORAS_ANTI_OVERFITTING.md**: Detalle completo de todas las mejoras
- **Instrrucciones.txt**: Requisitos del proyecto
- **Info adicional del dataset.txt**: Descripción de variables

## Contacto y Contribuciones

**Integrantes del equipo**:
- Integrante 1: [Nombre] - [Tareas realizadas]
- Integrante 2: [Nombre] - [Tareas realizadas]
- Integrante 3: [Nombre] - [Tareas realizadas]

---

**Fecha**: Noviembre 2025  
**Curso**: Machine Learning  
**Dataset**: UCI Gallstone Dataset
