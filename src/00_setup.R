# =============================================================================
# 00_setup.R
# Instalación de paquetes y configuración del entorno para el proyecto.
# Ejecutar este script una sola vez antes de correr el análisis principal.
# =============================================================================

# ----- Paquetes de R -----
paquetes <- c("reticulate", "tensorflow", "keras", "ggplot2", "caret", "dplyr")

paquetes_faltantes <- paquetes[!(paquetes %in% installed.packages()[, "Package"])]

if (length(paquetes_faltantes) > 0) {
  message("Instalando paquetes faltantes: ", paste(paquetes_faltantes, collapse = ", "))
  install.packages(paquetes_faltantes)
} else {
  message("Todos los paquetes de R ya están instalados.")
}

# ----- Configuración de TensorFlow y Python -----
library(reticulate)
library(tensorflow)
library(keras)

# Instalar TensorFlow si no está disponible
if (!reticulate::py_module_available("tensorflow")) {
  tensorflow::install_tensorflow()
}

# Instalar Pillow (necesario para el procesamiento de imágenes)
if (!reticulate::py_module_available("PIL")) {
  reticulate::py_install("Pillow")
}

# Verificar instalación
message("Versión de TensorFlow: ", tensorflow::tf_version())
tensorflow::tf_config()
