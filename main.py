from clases import VolumenDicom, ProcesadorVolumen

volumen = None
procesador = None

while True:
    print("\n===== MENU =====")
    print("1. Cargar carpeta DICOM (PPMI / Sarcoma / T2)")
    print("2. Corregir angulo")
    print("3. Ecualizar un corte")
    print("4. Segmentar un corte")
    print("5. Salir")
    opcion = input("Elige una opcion: ")

    try:
        if opcion == "1":
            nombre = input("Carpeta (PPMI, Sarcoma o T2): ")
            volumen = VolumenDicom("datos/" + nombre)
            procesador = ProcesadorVolumen(volumen)
            print("Paciente:", volumen.nombre_paciente)
            print("Modalidad:", volumen.modalidad)
            print("Filas x columnas:", volumen.filas, volumen.columnas)
            print("Pixel spacing:", volumen.pixel_spacing)
            print("Grosor de corte:", volumen.slice_thickness)
            print("Forma del volumen:", volumen.volumen.shape)

        elif opcion == "2":
            if volumen is None:
                print("Primero carga una carpeta (opcion 1).")
            else:
                grados = float(input("Angulo en grados: "))
                volumen.corregirAngulo(grados)
                print("Listo, volumen corregido.")

        elif opcion == "3":
            if volumen is None:
                print("Primero carga una carpeta (opcion 1).")
            else:
                print("Cortes disponibles: 0 a", volumen.volumen.shape[0] - 1)
                i = int(input("Numero de corte: "))
                procesador.ecualizar(i)

        elif opcion == "4":
            if volumen is None:
                print("Primero carga una carpeta (opcion 1).")
            else:
                plano = input("Plano (transversal, coronal, sagital): ")
                i = int(input("Indice del corte: "))
                umbral = int(input("Umbral (0-255, prueba 60): "))
                k = int(input("Tamano del kernel (ej. 5): "))
                it = int(input("Iteraciones (ej. 2): "))
                procesador.segmentar(plano, i, umbral, k, it)

        elif opcion == "5":
            break
        else:
            print("Opcion no valida.")

    except Exception as e:
        print("Error:", e)