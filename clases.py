
import os
import cv2
import numpy as np
import pydicom


class VolumenDicom:
    def __init__(self, carpeta, paciente=None):
        self.carpeta = carpeta
        self.paciente = paciente

        # Informacion del estudio
        self.nombre_paciente = None
        self.modalidad = None
        self.filas = None
        self.columnas = None
        self.pixel_spacing = None
        self.slice_thickness = None

        # Datos de los cortes
        self.datasets = []
        self.matrices_pixeles = []

        # Volumenes que construiremos despues
        self.volumen = None
        self.volumen_corregido = None

        self.cargar_carpeta()

    def cargar_carpeta(self):
        if not os.path.isdir(self.carpeta):
            raise FileNotFoundError(
                f"No existe la carpeta: {self.carpeta}"
            )

        
        archivos = []

        for raiz, carpetas, nombres_archivos in os.walk(self.carpeta):
            for nombre_archivo in nombres_archivos:
                ruta = os.path.join(raiz, nombre_archivo)

                try:
                    dataset = pydicom.dcmread(ruta)

                    if hasattr(dataset, "PixelData"):
                        archivos.append((ruta, dataset))

                except Exception:
                    continue


        if not archivos:
            raise ValueError(
                f"No se encontraron archivos DICOM con pixeles "
                f"en {self.carpeta}"
            )

        # Ordenar los cortes por su posicion espacial cuando sea posible
        if all(
            hasattr(ds, "ImagePositionPatient")
            for _, ds in archivos
        ):
            archivos.sort(
                key=lambda elemento:
                float(elemento[1].ImagePositionPatient[2])
            )
        elif all(
            hasattr(ds, "InstanceNumber")
            for _, ds in archivos
        ):
            archivos.sort(
                key=lambda elemento:
                int(elemento[1].InstanceNumber)
            )
        else:
            archivos.sort(key=lambda elemento: elemento[0])

        
        # Filtrar los archivos si se selecciono un paciente
        if self.paciente is not None:
            archivos = [
                (ruta, dataset)
                for ruta, dataset in archivos
                if str(getattr(dataset, "PatientID", "")) == str(self.paciente)
            ]

            if not archivos:
                raise ValueError(
                    f"No se encontraron archivos para el paciente {self.paciente}"
                )


        self.datasets = [ds for _, ds in archivos]

        # Extraer la informacion del primer corte
        primer_dataset = self.datasets[0]

        self.nombre_paciente = str(
            getattr(primer_dataset, "PatientName", "Desconocido")
        )
        self.modalidad = getattr(
            primer_dataset, "Modality", "Desconocida"
        )
        self.filas = int(primer_dataset.Rows)
        self.columnas = int(primer_dataset.Columns)

        self.pixel_spacing = tuple(
            float(valor)
            for valor in primer_dataset.PixelSpacing
        )
        
        # Espaciado efectivo despues de duplicar el tamaño
        self.pixel_spacing_escalado = tuple(
            valor / 2
            for valor in self.pixel_spacing
        )

        self.slice_thickness = float(
            primer_dataset.SliceThickness
        )

        # Extraer la matriz de pixeles de cada corte
        
        # Extraer y duplicar el tamaño de cada corte
        self.matrices_pixeles = []

        for dataset in self.datasets:
            matriz = dataset.pixel_array

            matriz_ampliada = cv2.resize(
                matriz,
                None,
                fx=2,
                fy=2,
                interpolation=cv2.INTER_LINEAR
            )

            self.matrices_pixeles.append(matriz_ampliada)
            self.volumen = np.stack(self.matrices_pixeles, axis=0)

    def corregirAngulo(self, grados):
        if self.volumen is None:
            raise ValueError("Primero debe construirse el volumen 3D")

        numero_cortes, filas, columnas = self.volumen.shape

        volumen_rotado = []

        centro = (columnas / 2, filas / 2)

        matriz_rotacion = cv2.getRotationMatrix2D(
            centro,
            grados,
            1.0
        )

        for corte in self.volumen:
            corte_rotado = cv2.warpAffine(
                corte,
                matriz_rotacion,
                (columnas, filas)
            )

            volumen_rotado.append(corte_rotado)

        self.volumen_corregido = np.stack(
            volumen_rotado,
            axis=0
        )