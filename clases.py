import os
import cv2
import numpy as np
import pydicom
import matplotlib.pyplot as plt


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
        self.pixel_spacing_escalado = None
        self.slice_thickness = None

        # Datos de los cortes
        self.datasets = []
        self.matrices_pixeles = []

        # Volumenes
        self.volumen = None
        self.volumen_corregido = None

        self.cargar_carpeta()

    def cargar_carpeta(self):
        if not os.path.isdir(self.carpeta):
            raise FileNotFoundError(f"No existe la carpeta: {self.carpeta}")

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
                f"No se encontraron archivos DICOM con pixeles en {self.carpeta}"
            )

        # Ordenar los cortes
        if all(hasattr(ds, "ImagePositionPatient") for _, ds in archivos):
            archivos.sort(key=lambda e: float(e[1].ImagePositionPatient[2]))
        elif all(hasattr(ds, "InstanceNumber") for _, ds in archivos):
            archivos.sort(key=lambda e: int(e[1].InstanceNumber))
        else:
            archivos.sort(key=lambda e: e[0])

        # Filtrar por paciente si se indico
        if self.paciente is not None:
            archivos = [
                (ruta, ds) for ruta, ds in archivos
                if str(getattr(ds, "PatientID", "")) == str(self.paciente)
            ]
            if not archivos:
                raise ValueError(
                    f"No se encontraron archivos para el paciente {self.paciente}"
                )

        self.datasets = [ds for _, ds in archivos]

        # Informacion del primer corte
        primero = self.datasets[0]
        self.nombre_paciente = str(getattr(primero, "PatientName", "Desconocido"))
        self.modalidad = getattr(primero, "Modality", "Desconocida")
        self.filas = int(primero.Rows)
        self.columnas = int(primero.Columns)
        self.pixel_spacing = tuple(float(v) for v in primero.PixelSpacing)
        self.pixel_spacing_escalado = tuple(v / 2 for v in self.pixel_spacing)
        self.slice_thickness = float(getattr(primero, "SliceThickness", 1.0))

        # Escalar cada corte al doble y construir el volumen 3D
        self.matrices_pixeles = []
        for dataset in self.datasets:
            matriz = dataset.pixel_array
            ampliada = cv2.resize(
                matriz, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR
            )
            self.matrices_pixeles.append(ampliada)

        self.volumen = np.stack(self.matrices_pixeles, axis=0)

    def corregirAngulo(self, grados):
        if self.volumen is None:
            raise ValueError("Primero debe construirse el volumen 3D")

        numero_cortes, filas, columnas = self.volumen.shape
        centro = (columnas / 2, filas / 2)
        matriz_rotacion = cv2.getRotationMatrix2D(centro, grados, 1.0)

        volumen_rotado = []
        for corte in self.volumen:
            corte_rotado = cv2.warpAffine(corte, matriz_rotacion, (columnas, filas))
            volumen_rotado.append(corte_rotado)

        self.volumen_corregido = np.stack(volumen_rotado, axis=0)


class ProcesadorVolumen:
    def __init__(self, volumen_dicom):
        self.vol = volumen_dicom
        self.volumen_copia = None
        self.corte_ecualizado = None
        self.corte_segmentado = None
        self.area_mm2 = None

    def _base(self):
        if self.vol.volumen_corregido is not None:
            return self.vol.volumen_corregido
        return self.vol.volumen

    @staticmethod
    def _a_uint8(matriz):
        m = matriz.astype(np.float32)
        m = cv2.normalize(m, None, 0, 255, cv2.NORM_MINMAX)
        return m.astype(np.uint8)

    # ---------------- ITEM 5: ECUALIZACION ----------------
    def ecualizar(self, indice_corte):
        self.volumen_copia = self._base().copy()

        if indice_corte < 0 or indice_corte >= self.volumen_copia.shape[0]:
            raise ValueError("Corte fuera de rango")

        corte_u8 = self._a_uint8(self.volumen_copia[indice_corte])
        self.corte_ecualizado = cv2.equalizeHist(corte_u8)

        fig, ejes = plt.subplots(2, 2, figsize=(10, 8))
        ejes[0, 0].imshow(corte_u8, cmap="gray")
        ejes[0, 0].set_title("Original (uint8)")
        ejes[0, 1].imshow(self.corte_ecualizado, cmap="gray")
        ejes[0, 1].set_title("Ecualizada")
        ejes[1, 0].hist(corte_u8.ravel(), bins=256, range=(0, 255))
        ejes[1, 0].set_title("Histograma original")
        ejes[1, 1].hist(self.corte_ecualizado.ravel(), bins=256, range=(0, 255))
        ejes[1, 1].set_title("Histograma ecualizado")
        plt.tight_layout()

        plt.savefig(f"grafico_ecualizacion_corte{indice_corte}.png")
        cv2.imwrite(f"imagen_ecualizada_corte{indice_corte}.png",
                    self.corte_ecualizado)
        plt.show()

    # ---------------- ITEM 6: SEGMENTACION ----------------
    def segmentar(self, plano, indice, umbral, tam_kernel, iteraciones):
        """plano: 'transversal', 'coronal' o 'sagital'"""
        volumen = self._base()
        n, filas, cols = volumen.shape
        ps = self.vol.pixel_spacing_escalado[0]      # mm por pixel (ya x2)
        grosor = self.vol.slice_thickness            # mm entre cortes

        # 1. Extraer el corte segun el plano
        if plano == "transversal":
            maximo = n
        elif plano == "coronal":
            maximo = filas
        elif plano == "sagital":
            maximo = cols
        else:
            raise ValueError("Plano invalido")

        if indice < 0 or indice >= maximo:
            raise ValueError("Indice fuera de rango")

        if plano == "transversal":
            corte = volumen[indice]
        elif plano == "coronal":
            corte = volumen[:, indice, :]
        else:
            corte = volumen[:, :, indice]

        # 2. Normalizar a uint8
        corte_u8 = self._a_uint8(corte)

        # 3. Corregir el aspect (en coronal/sagital el eje de cortes es mas grueso)
        if plano != "transversal":
            aspect = grosor / ps
            nuevo_alto = int(round(corte_u8.shape[0] * aspect))
            corte_u8 = cv2.resize(
                corte_u8, (corte_u8.shape[1], nuevo_alto),
                interpolation=cv2.INTER_LINEAR
            )
        # Ahora cada pixel mide ps x ps milimetros

        # 4. Binarizar: lo brillante (liquido/huecos en T2) queda en blanco
        _, binaria = cv2.threshold(corte_u8, umbral, 255, cv2.THRESH_BINARY)

        # 5. Apertura con kernel e iteraciones del usuario
        kernel = np.ones((tam_kernel, tam_kernel), np.uint8)
        abierta = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel,
                                   iterations=iteraciones)

        # 6. Componentes conectados
        num, etiquetas, stats, centroides = cv2.connectedComponentsWithStats(abierta)
        h, w = abierta.shape
        mejor = None
        mejor_area = 0
        for i in range(1, num):
            x, y, ancho, alto, area = stats[i]
            toca_borde = x == 0 or y == 0 or x + ancho >= w or y + alto >= h
            if not toca_borde and area > mejor_area:
                mejor, mejor_area = i, area
        if mejor is None and num > 1:
            mejor = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            mejor_area = stats[mejor, cv2.CC_STAT_AREA]

        # 7. Pasar a RGB y dibujar el circulo
        rgb = cv2.cvtColor(abierta, cv2.COLOR_GRAY2RGB)
        if mejor is not None:
            cx, cy = centroides[mejor]
            radio = int(max(np.sqrt(mejor_area / np.pi) * 1.5, 15))
            cv2.circle(rgb, (int(cx), int(cy)), radio, (255, 100, 0), 4)
            self.area_mm2 = mejor_area * ps * ps
        else:
            self.area_mm2 = 0.0

        self.corte_segmentado = rgb

        # 8. Graficar con titulos y dimensiones en mm
        ancho_mm = w * ps
        alto_mm = h * ps
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(rgb)
        ax.set_title(f"Segmentacion - plano {plano}, corte {indice}")
        ax.set_xlabel(f"Ancho: {ancho_mm:.1f} mm")
        ax.set_ylabel(f"Alto: {alto_mm:.1f} mm")
        ax.text(
            0.02, 0.02,
            f"Dimensiones: {ancho_mm:.1f} x {alto_mm:.1f} mm\n"
            f"Area elemento de interes: {self.area_mm2:.1f} mm2",
            transform=ax.transAxes, color="yellow", fontsize=10,
            bbox=dict(facecolor="black", alpha=0.6)
        )
        plt.tight_layout()
        plt.savefig(f"segmentacion_{plano}_{indice}.png")
        plt.show()

        print(f"Area aproximada: {self.area_mm2:.2f} mm2")
        return self.area_mm2