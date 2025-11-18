# Proyecto Final - Machine Learning: Clasificación de Gallstone Status

## Descripción del Proyecto

Este proyecto implementa modelos de clasificación con PyTorch para predecir la presencia o ausencia de cálculos biliares (Gallstone Status) basándose en variables demográficas, comorbilidades y mediciones corporales.

**Dataset**: UCI Gallstone Dataset  
**Variable objetivo**: Gallstone Status (0 = Present, 1 = Absent)  
**Modelos implementados**:
1. MLP (Multi-Layer Perceptron) optimizado con estrategias anti-overfitting
2. ResNet-style (Residual Network adaptado para datos tabulares)
3. MLP con hiperparámetros optimizados por Optuna
4. Logistic Regression con regularización Lasso (L1)
5. Logistic Regression con regularización Ridge (L2)
6. Logistic Regression con Elastic Net (L1 + L2)

## 🆕 Mejoras Implementadas (Noviembre 2025)

### Estrategias Anti-Overfitting
- ✅ **Data Augmentation con SMOTE**: Genera muestras sintéticas para balancear clases
- ✅ **Eliminación de features redundantes**: Reduce multicolinealidad (|r| > 0.8)
- ✅ **Arquitecturas más ligeras**: Menos parámetros → mejor generalización
- ✅ **Dropout aumentado**: De 0.3 a 0.5 para mayor regularización
- ✅ **Weight decay más fuerte**: L2 regularization de 1e-4
- ✅ **Class weights**: Compensa desbalances en función de pérdida
- ✅ **ReduceLROnPlateau**: Ajuste dinámico del learning rate
- ✅ **Gradient clipping**: Previene explosión de gradientes
- ✅ **Early stopping agresivo**: Patience reducido a 10 épocas

### Búsqueda de Hiperparámetros
- ✅ **Optuna**: Optimización automática de lr, dropout, hidden_dims, weight_decay, batch_size
- ✅ **30 trials** con pruning inteligente (MedianPruner)

### Modelos Adicionales
- ✅ **Lasso, Ridge, Elastic Net**: Baselines interpretables con feature selection
- ✅ **Feature importance analysis**: Identifica variables más predictivas

Ver detalles completos en [MEJORAS_ANTI_OVERFITTING.md](MEJORAS_ANTI_OVERFITTING.md)

## Estructura del Proyecto

```
ML_ProyectoFinal/
├── dataset-uci.csv                  # Dataset en formato CSV
├── dataset-uci.xlsx                 # Dataset original en Excel
├── Modelo2_Gallstone.ipynb          # Notebook principal con todo el código
├── Info adicional del dataset.txt  # Descripción de variables
├── Instrrucciones.txt              # Instrucciones del proyecto
├── README.md                        # Este archivo
├── mlp_classifier_best.pth         # Mejor modelo MLP guardado
└── resnet_classifier_best.pth      # Mejor modelo ResNet guardado
```

## Requisitos e Instalación

### Dependencias principales

```bash
pip install torch torchvision torchaudio
pip install pandas numpy matplotlib seaborn scikit-learn
pip install tqdm openpyxl
```

O instalar todas desde `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Archivo `requirements.txt`

```
torch>=2.0.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
tqdm>=4.65.0
openpyxl>=3.1.0
imbalanced-learn>=0.11.0  # Para SMOTE
optuna>=3.3.0             # Para búsqueda de hiperparámetros
plotly>=5.17.0            # Para visualizaciones de Optuna
```

## Uso del Notebook

1. **Abrir el notebook**: `Modelo2_Gallstone.ipynb` en Jupyter Lab o VS Code
2. **Ejecutar las celdas secuencialmente** desde la sección "CARGA DE DATOS"
3. **Secciones del notebook**:
   - **Carga de Datos**: Lee `dataset-uci.csv`
   - **EDA**: Análisis exploratorio con visualizaciones
   - **Preprocesamiento**: División train/val/test y estandarización
   - **PyTorch Setup**: Creación de DataLoaders
   - **Modelo 1 (MLP)**: Arquitectura, entrenamiento y evaluación
   - **Modelo 2 (ResNet)**: Arquitectura, entrenamiento y evaluación
   - **Comparativa**: Análisis comparativo de ambos modelos

## Características Implementadas

### Durante el Entrenamiento
- ✅ **Optimizador Adam** con weight decay fuerte (1e-4)
- ✅ **ReduceLROnPlateau scheduler** para ajuste dinámico de lr
- ✅ **Gradient clipping** (max_norm=1.0) para estabilidad
- ✅ **Class weights** en CrossEntropyLoss para balanceo
- ✅ **Early Stopping** con patience=10 épocas (más agresivo)
- ✅ **Guardado automático del mejor modelo** según `val_loss`
- ✅ **Barra de progreso** (tqdm) para visualizar avance de épocas
- ✅ **Reporte de métricas** en cada época: `train_loss`, `val_loss`, `train_acc`, `val_acc`, `train_f1`, `val_f1`, `lr`

### Evaluación y Visualización
- ✅ **Curvas de aprendizaje**: Loss, Accuracy y F1-Score
- ✅ **Análisis de overfitting/underfitting** automático
- ✅ **Reporte de clasificación detallado** por clase (precision, recall, F1-score)
- ✅ **Matriz de confusión** con visualización
- ✅ **Comparativa final** entre modelos

## Arquitecturas de Modelos

### MLP Classifier Optimizado
**ANTES** (sobreajustado):
- Input → Linear(256) → BatchNorm → ReLU → Dropout(0.3)
- → Linear(128) → BatchNorm → ReLU → Dropout(0.3)
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
