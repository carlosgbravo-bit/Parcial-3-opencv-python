
import cv2
import numpy as np

from clases import VolumenDicom

print("Proyecto de OpenCV iniciado")

print("Versión de OpenCV:", cv2.__version__)
print("Versión de NumPy:", np.__version__)

print("\n--- Prueba de archivos DICOM ---")
import os

print("\n--- Inspección de PPMI ---")

for raiz, carpetas, archivos in os.walk("datos/PPMI"):
    print("Carpeta:", raiz)
    print("Subcarpetas:", carpetas)
    print("Cantidad de archivos:", len(archivos))
    print("Primeros archivos:", archivos[:5])
    print("---")
estudio = VolumenDicom("datos/PPMI", paciente="3105")

print("Nombre del paciente:", estudio.nombre_paciente)
print("Modalidad:", estudio.modalidad)
print("Filas:", estudio.filas)
print("Columnas:", estudio.columnas)
print("Pixel spacing:", estudio.pixel_spacing)
print("Espesor del corte:", estudio.slice_thickness)
print("Cantidad de cortes:", len(estudio.matrices_pixeles))

print(
    "Dimensiones de la primera matriz:",
    estudio.matrices_pixeles[0].shape
)
print("\n--- Identificación de estudios ---")

for dataset in estudio.datasets:
    print(
        "Paciente:", getattr(dataset, "PatientID", "Desconocido"),
        "| Estudio:", getattr(dataset, "StudyInstanceUID", "Desconocido"),
        "| Serie:", getattr(dataset, "SeriesInstanceUID", "Desconocido")
    )
    break


print("\n--- Pacientes encontrados ---")

pacientes = set()

for dataset in estudio.datasets:
    pacientes.add(str(getattr(dataset, "PatientID", "Desconocido")))

print("Cantidad de pacientes:", len(pacientes))
print("Identificadores:", sorted(pacientes))
