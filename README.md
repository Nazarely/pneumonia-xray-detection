# 🫁 Detección de Neumonía en Radiografías de Tórax

Proyecto de **Deep Learning** en R para clasificar radiografías de tórax como **NORMAL** o **PNEUMONIA**, utilizando Transfer Learning con el modelo **Xception** preentrenado en ImageNet.

---

## 📌 Descripción del problema

La neumonía es una infección pulmonar que puede ser difícil de detectar sin herramientas adecuadas. Este proyecto entrena un modelo de red neuronal convolucional profunda capaz de **analizar automáticamente** radiografías de tórax y clasificarlas con alta precisión, lo que podría asistir en el diagnóstico clínico temprano.

**Tipo de problema:** Clasificación binaria de imágenes

| Categoría     | Descripción                            |
|---------------|----------------------------------------|
| 🟢 NORMAL     | Radiografía sin indicios de neumonía   |
| 🔴 PNEUMONIA  | Radiografía con indicios de neumonía   |

---

## 📂 Estructura del repositorio

```
pneumonia-xray-detection/
│
├── data/
│   └── chest_xray/          # Dataset (ver instrucciones abajo)
│       ├── train/
│       │   ├── NORMAL/
│       │   └── PNEUMONIA/
│       └── test/
│           ├── NORMAL/
│           └── PNEUMONIA/
│
├── src/
│   ├── 00_setup.R           # Instalación de paquetes y configuración del entorno
│   └── 01_deteccion_neumonia.R  # Script principal: modelo, entrenamiento y evaluación
│
├── outputs/
│   ├── accuracy_por_epoca.png
│   ├── loss_por_epoca.png
│   ├── matriz_confusion_test.png
│   └── modelo_deteccion_neumonia.keras
│
├── .gitignore
└── README.md
```

---

## 📊 Dataset

- **Nombre:** Chest X-Ray Images (Pneumonia)
- **Fuente:** [Kaggle — Paul Mooney](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Imágenes:** ~5,800 radiografías de tórax en formato JPEG
- **División:** train / test (organizadas por carpetas por clase)

> ⚠️ El dataset no está incluido en el repositorio por su tamaño (~1.2 GB). Descargarlo desde Kaggle y colocarlo en `data/chest_xray/` respetando la estructura de carpetas original.

---

## 🧠 Arquitectura del modelo

Se utiliza **Transfer Learning** sobre **Xception**, una CNN profunda preentrenada en ImageNet con más de 22 millones de parámetros.

```
Entrada (224 × 224 × 3)
        │
   Xception base        ← Pesos congelados (ImageNet)
        │
GlobalAveragePooling2D
        │
  Dense(1024, ReLU)     ← Capas entrenables
        │
   Dropout(0.2)
        │
  Dense(1, Sigmoid)     ← Salida: P(Neumonía)
```

**Ventajas del Transfer Learning:**
- Se aprovechan características visuales aprendidas en millones de imágenes
- Se necesitan muchos menos datos para alcanzar buena performance
- El entrenamiento es significativamente más rápido

---

## ⚙️ Metodología

```
Imágenes de rayos X
        │
        ▼
Preprocesamiento (resize 224×224, normalización, aumentación)
        │
        ▼
   División: 80% train / 20% validación / test separado
        │
        ▼
  Transfer Learning con Xception (pesos ImageNet congelados)
        │
        ▼
    Fine-tuning de capas densas (10 épocas, Adam, lr=0.001)
        │
        ▼
  Evaluación: Accuracy, Loss, Matriz de Confusión
```

---

## 📈 Resultados

> Los valores se obtienen al ejecutar el script con los datos originales.

| Métrica          | Test Set |
|------------------|----------|
| Accuracy         | —        |
| Loss             | —        |
| Sensibilidad     | —        |
| Especificidad    | —        |


### Curvas de entrenamiento

| Accuracy por época | Loss por época |
|---|---|
| *(generado al correr el script)* | *(generado al correr el script)* |

---

## 🚀 Cómo ejecutar el proyecto

### 1. Clonar el repositorio
```bash
git clone https://github.com/tu-usuario/pneumonia-xray-detection.git
cd pneumonia-xray-detection
```

### 2. Descargar el dataset
Descargarlo desde [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) y colocarlo en:
```
data/chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

### 3. Instalar dependencias
Ejecutar en R o RStudio (una sola vez):
```r
source("src/00_setup.R")
```

### 4. Entrenar y evaluar el modelo
```r
source("src/01_deteccion_neumonia.R")
```

### 5. Predecir sobre una nueva imagen
```r
resultado <- predecir_imagen("ruta/a/tu/imagen.jpeg")
```

---

## 🛠️ Tecnologías y paquetes

![R](https://img.shields.io/badge/R-276DC3?style=for-the-badge&logo=r&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)

| Paquete        | Uso                                               |
|----------------|---------------------------------------------------|
| `keras`        | Construcción y entrenamiento del modelo           |
| `tensorflow`   | Backend de cómputo                                |
| `reticulate`   | Interfaz R–Python para TensorFlow                 |
| `ggplot2`      | Visualización de curvas de entrenamiento          |
| `caret`        | Matriz de confusión y métricas de evaluación      |

---

## 👤 Autor

**[Nazarely Gomez Abularach]**
T.S en Ciencia de Datos e Inteligencia Artificial

[![LinkedIn](www.linkedin.com/in/nazarely-gomez-abularach)
[![GitHub](https://github.com/Nazarely)

---

## 📄 Licencia

Este proyecto está bajo la licencia MIT. Ver el archivo [LICENSE](LICENSE) para más detalles.
