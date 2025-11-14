# Proyecto Final - Machine Learning: Clasificación de Gallstone Status

## Descripción del Proyecto

Este proyecto implementa modelos de clasificación con PyTorch para predecir la presencia o ausencia de cálculos biliares (Gallstone Status) basándose en variables demográficas, comorbilidades y mediciones corporales.

**Dataset**: UCI Gallstone Dataset  
**Variable objetivo**: Gallstone Status (0 = Present, 1 = Absent)  
**Modelos implementados**:
1. MLP (Multi-Layer Perceptron) con Batch Normalization y Dropout
2. ResNet-style (Residual Network adaptado para datos tabulares)

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
- ✅ **Optimizador Adam** con weight decay (regularización L2)
- ✅ **Early Stopping** con patience=15 épocas
- ✅ **Guardado automático del mejor modelo** según `val_loss`
- ✅ **Barra de progreso** (tqdm) para visualizar avance de épocas
- ✅ **Reporte de métricas** en cada época: `train_loss`, `val_loss`, `train_acc`, `val_acc`, `train_f1`, `val_f1`

### Evaluación y Visualización
- ✅ **Curvas de aprendizaje**: Loss, Accuracy y F1-Score
- ✅ **Análisis de overfitting/underfitting** automático
- ✅ **Reporte de clasificación detallado** por clase (precision, recall, F1-score)
- ✅ **Matriz de confusión** con visualización
- ✅ **Comparativa final** entre modelos

## Arquitecturas de Modelos

### MLP Classifier
- Input → Linear(256) → BatchNorm → ReLU → Dropout(0.3)
- → Linear(128) → BatchNorm → ReLU → Dropout(0.3)
- → Linear(64) → BatchNorm → ReLU → Dropout(0.3)
- → Linear(2) [Output]

### ResNet Classifier
- Input → Linear(256) → BatchNorm → ReLU → Dropout(0.3) [Proyección]
- → 4x Bloques Residuales [256 → 256 con skip connections]
- → Linear(2) [Clasificador]

**Bloque Residual**:
```
x → Linear(256) → BatchNorm → ReLU → Dropout → Linear(256) → BatchNorm → (+) → ReLU
↑_______________________________________________________________|
```

## Resultados Esperados

- **Accuracy**: >85% en conjunto de prueba
- **F1-Score**: >0.85 (weighted)
- **Curvas**: Convergencia sin overfitting significativo
- **Tiempo de entrenamiento**: ~2-5 minutos por modelo (CPU), <1 min (GPU)

## Ejecución Rápida

```python
# En una celda nueva del notebook:
# Cargar el mejor modelo guardado
model_mlp.load_state_dict(torch.load('mlp_classifier_best.pth'))
model_mlp.eval()

# Hacer predicciones
with torch.no_grad():
    outputs = model_mlp(X_test_tensor.to(device))
    _, preds = torch.max(outputs, 1)
```

## Notas Importantes

1. **GPU vs CPU**: El código detecta automáticamente si CUDA está disponible
2. **Reproducibilidad**: Se utilizan `random_state=42` en todas las divisiones
3. **Estratificación**: Los conjuntos train/val/test mantienen la proporción de clases
4. **Normalización**: StandardScaler ajustado solo en train, transformado en val/test

## Mejoras Futuras (Opcionales)

- Implementar búsqueda de hiperparámetros con Optuna
- Probar modelos adicionales: CatBoost, XGBoost, TabNet
- Selección de features con algoritmos genéticos
- Cross-validation k-fold
- Ensemble methods (voting, stacking)

## Contacto y Contribuciones

**Integrantes del equipo**:
- Integrante 1: [Nombre] - [Tareas realizadas]
- Integrante 2: [Nombre] - [Tareas realizadas]
- Integrante 3: [Nombre] - [Tareas realizadas]

---

**Fecha**: Noviembre 2025  
**Curso**: Machine Learning  
**Dataset**: UCI Gallstone Dataset
