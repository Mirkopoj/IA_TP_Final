import sys

import pyaudio

from estacionamiento.entrada import Entrada

# Desactivar __pycache__
sys.dont_write_bytecode = True

device_index = 2
audio = pyaudio.PyAudio()

print(
    "----------------------" + "Lista de dispositivos de entrada---------------------"
)
info = audio.get_host_api_info_by_index(0)
info_cantidad = info.get("deviceCount")
numero_dispositivos = int(info_cantidad) if info_cantidad is not None else 0

for i in range(0, numero_dispositivos):
    canales = audio.get_device_info_by_host_api_device_index(0, i).get(
        "maxInputChannels", 0
    )
    if int(canales) > 0:
        print(
            f'Id dispositivo {i} - {audio.get_device_info_by_host_api_device_index(0, i).get("name")}'
        )

print(
    "--------------------------------------" + "-------------------------------------"
)

idx = int(input().strip())
print(f"Se tomara el audio con el dispositivo {idx}")
entrada = Entrada(audio, idx)

while True:
    print("Esperando llegada de un auto")
    input()

    entrada.informar_mensaje("Bienvenido al estacionamiento!")

    patente: str = entrada.adquirir_patente()

    reserva = entrada.tiene_reserva(patente)

    if reserva:
        entrada.permitir_estacionar(reserva)
    else:
        entrada.informar_mensaje("No se encontro reserva")

        estadia = entrada.adquirir_estadia()

        techado = entrada.adquirir_techado()

        reserva = entrada.verificar_espacios(patente, estadia, techado)

        if reserva:
            entrada.permitir_estacionar(reserva)
        else:
            entrada.informar_mensaje(
                "No hay espacio disponible, retire su vehiculo por favor"
            )
