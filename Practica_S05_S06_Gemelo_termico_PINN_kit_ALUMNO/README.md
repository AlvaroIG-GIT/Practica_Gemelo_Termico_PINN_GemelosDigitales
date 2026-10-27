# Práctica «Gemelo térmico de una barra con PINN» · material del alumno
**Gemelos Digitales · UIE · Unidad II · Sesión 05**

Todo funciona en un PC convencional, **sin GPU y sin hardware**. Ningún entrenamiento pasa de 3 minutos.

## Instalación (5 minutos)

```bash
pip install torch numpy pandas matplotlib      # paho-mqtt solo si usáis el broker del curso
jupyter lab Practica_gemelo_termico.ipynb      # o abridlo con VS Code
```

En **Google Colab**: subid este zip y ejecutad la primera celda del cuaderno, que lo descomprime.

## Qué hay aquí

| Fichero | Para quién | Qué es |
|---|---|---|
| `Practica_gemelo_termico.ipynb` | Alumno | El cuaderno de la práctica. Las celdas marcadas las escribe el alumno; las demás se ejecutan tal cual. |
| `docs/` | Alumno | Ficha de la práctica y material de apoyo sobre la ecuación del calor (Word). |
| `pinn_barra.py` | — | El módulo de la práctica: datos adimensionales, PINN con rasgos de Fourier, residuo de la EDP, entrenamiento Adam→L-BFGS, sensor virtual, error por ventanas, guardar/cargar. |
| `planta_virtual.py` | — | **Vuestra barra.** Simula el ensayo y guarda el CSV con el formato del montaje real. |
| `gen_datos.py` | — | La física de la planta (diferencias finitas) y el generador de los ensayos de `data/`. |
| `data/` | — | Ensayos de referencia genéricos, por si no podéis generar el vuestro. |
| `opcional_hardware/` | — | Firmware ESP32, flujo de Node-RED, `docker-compose.yml` y exportador de InfluxDB, por si se monta la barra real. **No hace falta para la práctica.** |

## La planta virtual

```bash
python planta_virtual.py --pareja p07                      # ensayo base      -> data/p07_base.csv
python planta_virtual.py --pareja p07 --ensayo ventilador  # deriva (reto)    -> data/p07_ventilador.csv
python planta_virtual.py --pareja p07 --ensayo aislante    # deriva, variante
python planta_virtual.py --pareja p07 --misterio           # otro metal (súper reto)
```

Cada pareja tiene su propia barra: α y h se derivan del identificador, así que son distintos para cada
una y estables entre ejecuciones. Usad siempre el identificador que os haya asignado el profesor.

## Formato de los ensayos

CSV con columnas `t_s, S1, S2, S3, S4, S5, S6`, 451 filas (una lectura cada 2 s durante 900 s).
Los sensores S1 a S5 están en x = 3, 9, 15, 21 y 27 cm. **S6 (x = 1 cm) es el sensor oculto**: el módulo
nunca lo usa para entrenar; solo `error_sensor()` lo lee para validar el sensor virtual.
