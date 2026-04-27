# =============================================================================
# 01_deteccion_neumonia.R
#
# Detección de Neumonía en Radiografías de Tórax con Deep Learning
#
# Objetivo: Clasificar imágenes de rayos X de tórax en dos categorías —
# NORMAL y PNEUMONIA — usando Transfer Learning con el modelo Xception
# preentrenado en ImageNet.
#
# Dataset: Chest X-Ray Images (Pneumonia)
# Fuente:  Kaggle — Paul Mooney
# Link:    https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
#
# Arquitectura: Xception (preentrenado) + capas densas personalizadas
# Tipo de problema: Clasificación binaria
#
# Autor: [Tu nombre]
# Fecha: [Fecha]
# =============================================================================


# -----------------------------------------------------------------------------
# 1. LIBRERÍAS
# -----------------------------------------------------------------------------
# Si algo no está instalado, ejecutar primero: source("src/00_setup.R")

library(reticulate)
library(tensorflow)
library(keras)
library(ggplot2)
library(dplyr)
library(caret)


# -----------------------------------------------------------------------------
# 2. CONFIGURACIÓN GENERAL
# -----------------------------------------------------------------------------

# Semilla para reproducibilidad
set.seed(2021)
tf$random$set_seed(2021L)

# Parámetros de imagen
WIDTH       <- 224L
HEIGHT      <- 224L
CHANNELS    <- 3L
TARGET_SIZE <- c(WIDTH, HEIGHT)
INPUT_SHAPE <- as.integer(c(WIDTH, HEIGHT, CHANNELS))

# Parámetros de entrenamiento
BATCH_SIZE  <- 32L
EPOCHS      <- 10L

# Etiquetas de clasificación
LABEL_LIST  <- c("NORMAL", "PNEUMONIA")
OUTPUT_N    <- length(LABEL_LIST)

# Directorios de datos (rutas relativas desde la raíz del proyecto)
PATH_TRAIN  <- "data/chest_xray/train/"
PATH_TEST   <- "data/chest_xray/test/"
PATH_OUTPUT <- "outputs/"

# Crear directorio de outputs si no existe
if (!dir.exists(PATH_OUTPUT)) dir.create(PATH_OUTPUT, recursive = TRUE)

cat("Configuración cargada correctamente.\n")
cat("Clases:", paste(LABEL_LIST, collapse = " | "), "\n")


# -----------------------------------------------------------------------------
# 3. CARGA Y PREPARACIÓN DE DATOS
# -----------------------------------------------------------------------------

# Generador de entrenamiento con aumentación de datos leve
# y 20% reservado para validación
train_data_gen <- image_data_generator(
  rescale          = 1 / 255,
  validation_split = 0.2,
  horizontal_flip  = TRUE,   # Espejado horizontal (aumentación)
  zoom_range       = 0.1     # Zoom leve (aumentación)
)

# Generador de test (solo normalización, sin aumentación)
test_data_gen <- image_data_generator(rescale = 1 / 255)

# Cargamos imágenes de entrenamiento
train_images <- flow_images_from_directory(
  directory   = PATH_TRAIN,
  generator   = train_data_gen,
  subset      = "training",
  target_size = TARGET_SIZE,
  class_mode  = "binary",
  batch_size  = BATCH_SIZE,
  classes     = LABEL_LIST,
  shuffle     = TRUE,
  seed        = 2021
)

# Cargamos imágenes de validación
validation_images <- flow_images_from_directory(
  directory   = PATH_TRAIN,
  generator   = train_data_gen,
  subset      = "validation",
  target_size = TARGET_SIZE,
  class_mode  = "binary",
  batch_size  = BATCH_SIZE,
  classes     = LABEL_LIST,
  seed        = 2021
)

# Cargamos imágenes de test
test_images <- flow_images_from_directory(
  directory   = PATH_TEST,
  generator   = test_data_gen,
  target_size = TARGET_SIZE,
  class_mode  = "binary",
  batch_size  = BATCH_SIZE,
  shuffle     = FALSE     # Importante: no mezclar para que las predicciones sean comparables
)

cat("Imágenes de entrenamiento:", train_images$n, "\n")
cat("Imágenes de validación:   ", validation_images$n, "\n")
cat("Imágenes de test:         ", test_images$n, "\n")


# -----------------------------------------------------------------------------
# 4. ARQUITECTURA DEL MODELO (Transfer Learning — Xception)
# -----------------------------------------------------------------------------

# Entrada
input <- tf$keras$Input(shape = INPUT_SHAPE)

# Modelo base Xception preentrenado en ImageNet (sin capas densas finales)
mod_base <- application_xception(
  weights     = "imagenet",
  include_top = FALSE,
  input_shape = INPUT_SHAPE
)

# Congelamos los pesos del modelo base para no reentrenar Xception
freeze_weights(mod_base)

# Capas personalizadas encima del modelo base
# GlobalAveragePooling2D reduce los mapas de características a un vector 1D
# Dense(1024) aprende representaciones específicas de nuestro dataset
# Dropout(0.2) regulariza para evitar overfitting
# Dense(1, sigmoid) salida binaria: probabilidad de neumonía
output <- mod_base(input) %>%
  tf$keras$layers$GlobalAveragePooling2D()() %>%
  tf$keras$layers$Dense(units = 1024L, activation = "relu")() %>%
  tf$keras$layers$Dropout(rate = 0.2)() %>%
  tf$keras$layers$Dense(units = 1L,    activation = "sigmoid")()

# Construcción del modelo funcional
model <- tf$keras$Model(inputs = input, outputs = output)

# Compilación
model$compile(
  loss      = "binary_crossentropy",
  optimizer = tf$keras$optimizers$Adam(learning_rate = 0.001),
  metrics   = list("accuracy")
)

# Resumen de la arquitectura
model$summary()


# -----------------------------------------------------------------------------
# 5. ENTRENAMIENTO
# -----------------------------------------------------------------------------

steps_per_epoch  <- as.integer(train_images$n %/% BATCH_SIZE)
validation_steps <- as.integer(validation_images$n %/% BATCH_SIZE)

cat("\nIniciando entrenamiento...\n")
cat("Épocas:", EPOCHS, "| Batch size:", BATCH_SIZE, "\n\n")

hist <- model$fit(
  x                = train_images,
  steps_per_epoch  = steps_per_epoch,
  epochs           = as.integer(EPOCHS),
  validation_data  = validation_images,
  validation_steps = validation_steps,
  verbose          = 1
)


# -----------------------------------------------------------------------------
# 6. VISUALIZACIÓN DEL ENTRENAMIENTO
# -----------------------------------------------------------------------------

# Convertimos el historial a un dataframe para graficarlo con ggplot2
historia_df <- data.frame(
  Epoca           = seq_len(EPOCHS),
  Train_Accuracy  = hist$history$accuracy,
  Val_Accuracy    = hist$history$val_accuracy,
  Train_Loss      = hist$history$loss,
  Val_Loss        = hist$history$val_loss
)

# Gráfico de Accuracy por época
plot_accuracy <- ggplot(historia_df, aes(x = Epoca)) +
  geom_line(aes(y = Train_Accuracy, color = "Entrenamiento"), linewidth = 1) +
  geom_line(aes(y = Val_Accuracy,   color = "Validación"),    linewidth = 1, linetype = "dashed") +
  geom_point(aes(y = Train_Accuracy, color = "Entrenamiento"), size = 2) +
  geom_point(aes(y = Val_Accuracy,   color = "Validación"),    size = 2) +
  scale_color_manual(values = c("Entrenamiento" = "#1976D2", "Validación" = "#E53935")) +
  scale_x_continuous(breaks = seq_len(EPOCHS)) +
  labs(
    title    = "Accuracy por Época — Detección de Neumonía",
    subtitle = "Transfer Learning con Xception",
    x        = "Época",
    y        = "Accuracy",
    color    = NULL
  ) +
  theme_minimal() +
  theme(legend.position = "bottom")

print(plot_accuracy)
ggsave(file.path(PATH_OUTPUT, "accuracy_por_epoca.png"), plot_accuracy, width = 8, height = 5, dpi = 150)

# Gráfico de Loss por época
plot_loss <- ggplot(historia_df, aes(x = Epoca)) +
  geom_line(aes(y = Train_Loss, color = "Entrenamiento"), linewidth = 1) +
  geom_line(aes(y = Val_Loss,   color = "Validación"),    linewidth = 1, linetype = "dashed") +
  geom_point(aes(y = Train_Loss, color = "Entrenamiento"), size = 2) +
  geom_point(aes(y = Val_Loss,   color = "Validación"),    size = 2) +
  scale_color_manual(values = c("Entrenamiento" = "#1976D2", "Validación" = "#E53935")) +
  scale_x_continuous(breaks = seq_len(EPOCHS)) +
  labs(
    title    = "Loss por Época — Detección de Neumonía",
    subtitle = "Transfer Learning con Xception",
    x        = "Época",
    y        = "Loss (Binary Crossentropy)",
    color    = NULL
  ) +
  theme_minimal() +
  theme(legend.position = "bottom")

print(plot_loss)
ggsave(file.path(PATH_OUTPUT, "loss_por_epoca.png"), plot_loss, width = 8, height = 5, dpi = 150)


# -----------------------------------------------------------------------------
# 7. EVALUACIÓN EN EL SET DE TEST
# -----------------------------------------------------------------------------

cat("\nEvaluando modelo en set de test...\n")

score <- model$evaluate(
  test_images,
  steps   = as.integer(test_images$n %/% BATCH_SIZE),
  verbose = 1
)

cat("\n--- Resultados en Test ---\n")
cat("Loss:    ", round(score[[1]], 4), "\n")
cat("Accuracy:", round(score[[2]], 4), "\n")


# -----------------------------------------------------------------------------
# 8. MATRIZ DE CONFUSIÓN EN TEST
# -----------------------------------------------------------------------------

# Obtenemos predicciones sobre todo el set de test
predicciones_prob <- model$predict(
  test_images,
  steps = as.integer(ceiling(test_images$n / BATCH_SIZE))
)

# Convertimos probabilidades a etiquetas binarias (umbral = 0.5)
predicciones_clase <- ifelse(predicciones_prob > 0.5, 1, 0)

# Etiquetas reales
etiquetas_reales <- test_images$classes[seq_len(length(predicciones_clase))]

# Calculamos la matriz de confusión
cm <- confusionMatrix(
  data      = factor(predicciones_clase, levels = c(0, 1), labels = LABEL_LIST),
  reference = factor(etiquetas_reales,   levels = c(0, 1), labels = LABEL_LIST)
)
print(cm)

# Graficamos la matriz de confusión
cm_df <- as.data.frame(cm$table)

plot_cm <- ggplot(cm_df, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), size = 6, fontface = "bold") +
  scale_fill_gradient(low = "white", high = "#1976D2") +
  labs(
    title    = "Matriz de Confusión — Set de Test",
    subtitle = paste0("Accuracy: ", round(cm$overall["Accuracy"] * 100, 2), "%"),
    x        = "Valor Verdadero",
    y        = "Predicción",
    fill     = "Frecuencia"
  ) +
  theme_minimal()

print(plot_cm)
ggsave(file.path(PATH_OUTPUT, "matriz_confusion_test.png"), plot_cm, width = 6, height = 5, dpi = 150)


# -----------------------------------------------------------------------------
# 9. FUNCIÓN DE PREDICCIÓN PARA NUEVAS IMÁGENES
# -----------------------------------------------------------------------------

# Esta función recibe la ruta a una imagen de rayos X y devuelve
# si el modelo detecta neumonía o no.

predecir_imagen <- function(ruta_imagen, umbral = 0.5) {
  # Verificar que el archivo existe
  if (!file.exists(ruta_imagen)) {
    stop("No se encontró el archivo: ", ruta_imagen)
  }

  # Cargar y preprocesar la imagen
  imagen <- tf$image$decode_image(
    tf$io$read_file(ruta_imagen),
    channels = 1L   # grayscale
  )

  target_tensor <- tf$constant(c(224L, 224L), dtype = "int32")
  imagen <- tf$image$resize(imagen, size = target_tensor)
  imagen <- tf$image$grayscale_to_rgb(imagen)  # Xception espera 3 canales
  imagen <- imagen / 255                        # Normalización
  imagen <- tf$expand_dims(imagen, axis = 0L)   # Agregar dimensión de batch

  # Predicción
  prob <- model$predict(imagen, verbose = 0)[[1]]

  # Resultado
  clase     <- ifelse(prob > umbral, "PNEUMONIA", "NORMAL")
  confianza <- ifelse(prob > umbral, prob, 1 - prob) * 100

  cat("-----------------------------------\n")
  cat("Imagen analizada:", basename(ruta_imagen), "\n")
  cat("Diagnóstico:     ", clase, "\n")
  cat("Confianza:       ", round(confianza, 2), "%\n")
  cat("-----------------------------------\n")

  return(list(clase = clase, probabilidad = prob, confianza = confianza))
}

# Ejemplo de uso:
# resultado <- predecir_imagen("data/chest_xray/test/PNEUMONIA/alguna_imagen.jpeg")


# -----------------------------------------------------------------------------
# 10. GUARDAR EL MODELO
# -----------------------------------------------------------------------------

# Guardamos en formato .keras (recomendado en versiones modernas de Keras)
modelo_path <- file.path(PATH_OUTPUT, "modelo_deteccion_neumonia.keras")
model$save(modelo_path)
cat("Modelo guardado en:", modelo_path, "\n")

# Para cargar el modelo en el futuro:
# model <- keras::load_model_tf("outputs/modelo_deteccion_neumonia.keras")
