
import cv2
import numpy as np

from clases import VolumenDicom

print("Proyecto de OpenCV iniciado")

print("Versión de OpenCV:", cv2.__version__)
print("Versión de NumPy:", np.__version__)

print("\n--- Prueba de archivos DICOM ---")

estudio = VolumenDicom("datos/PPMI", paciente="3105")

print("Nombre del paciente:", estudio.nombre_paciente)
print("Modalidad:", estudio.modalidad)
print("Filas:", estudio.filas)
print("Columnas:", estudio.columnas)
print("Pixel spacing:", estudio.pixel_spacing)
print("Pixel spacing escalado:", estudio.pixel_spacing_escalado)
print("Espesor del corte:", estudio.slice_thickness)
print("Cantidad de cortes:", len(estudio.matrices_pixeles))

print(
    "Dimensiones de la primera matriz:",
    estudio.matrices_pixeles[0].shape
)
print("Dimensiones del volumen 3D:", estudio.volumen.shape)
print("Tipo de dato del volumen:", estudio.volumen.dtype)

estudio.corregirAngulo(15)

print("Dimensiones del volumen corregido:", estudio.volumen_corregido.shape)
print("Dimensiones del volumen original:", estudio.volumen.shape)