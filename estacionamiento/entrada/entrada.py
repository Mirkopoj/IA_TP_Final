import datetime as dt
from typing import Optional

from pyaudio import PyAudio

from estacionamiento.asistente import Asistente
from estacionamiento.db import BaseDatos, Reserva, Tiempo
from estacionamiento.ia import (
    IA,
    EstadiaProcesamientoException,
    PatenteProcesamientoException,
)
from estacionamiento.ia.ia import ConfirmacionProcesamientoException


class Entrada:
    """
    Esta clase maneja la obtencion de la informacion mediante voz
    """

    def __init__(self, audio: PyAudio, id_dispositivo: int):
        self.asistente = Asistente()
        self.db = BaseDatos()
        self.ia = IA(audio, id_dispositivo)

    def obtener_patente(self) -> str:
        mensaje = "¿Podría decirme su patente por favor?"
        self.informar_mensaje(mensaje)
        return self.ia.procesar_patente()

    def obtener_estadia(self) -> int:
        mensaje = "¿Cuanto tiempo desea estacionar su vehiculo?"
        self.informar_mensaje(mensaje)
        return self.ia.procesar_estadia()

    def obtener_confirmacion(self, mensaje: str) -> bool:
        self.informar_mensaje(mensaje)
        return self.ia.procesar_confirmacion()

    def verificar_espacios(
        self, patente: str, estadia: int, techado: bool
    ) -> Optional[Reserva]:
        fecha = dt.datetime.now()
        tiempo = Tiempo(fecha=fecha, estadia=estadia)
        return self.db.verificar_espacios(patente, tiempo, techado)

    def tiene_reserva(self, patente: str) -> Optional[Reserva]:
        return self.db.tiene_reserva(patente)

    def permitir_estacionar(self, reserva: Reserva):
        self.asistente.habla(f"Estacione en la posición {reserva.posicion}")
        self.db.registrar_vehiculo(reserva)

    def informar_mensaje(self, mensaje: str):
        print(mensaje)
        self.asistente.habla(mensaje)

    def adquirir_patente(self) -> str:
        patente_ok = False
        confirmacion = False
        patente: str = ""
        while not patente_ok:
            try:
                patente = self.obtener_patente()
                patente_ok = True
                confirmacion = False
                while not confirmacion:
                    try:
                        mensaje = f"Entendí {patente} ¿Es esto correcto?"
                        confirmacion = self.obtener_confirmacion(mensaje)
                        if not confirmacion:
                            self.informar_mensaje(
                                "Vuelva a indicar su patente por favor"
                            )
                            patente_ok = False
                            confirmacion = True
                        else:
                            self.informar_mensaje(f"Patente {patente} confirmada!")
                    except ConfirmacionProcesamientoException:
                        self.informar_mensaje(
                            "No se pudo procesar la confirmacion,"
                            + " intente nuevamente"
                        )
            except PatenteProcesamientoException:
                self.informar_mensaje(
                    "No se pudo procesar la patente, intente nuevamente"
                )
        return patente

    def adquirir_estadia(self) -> int:
        estadia_ok = False
        estadia: int = 0
        while not estadia_ok:
            try:
                estadia = self.obtener_estadia()
                estadia_ok = True
                confirmacion = False
                while not confirmacion:
                    try:
                        horas = estadia // 60
                        minutos = estadia % 60
                        horas_s = "una hora" if horas == 1 else f"{horas} horas"
                        minutos_s = (
                            "un minuto" if minutos == 1 else f"{minutos} minutos"
                        )
                        mensaje = f"Entendí {horas_s} {minutos_s} ¿Es esto correcto?"
                        confirmacion = self.obtener_confirmacion(mensaje)
                        if not confirmacion:
                            self.informar_mensaje(
                                "Por favor vuelva a indicar su estadia"
                            )
                            estadia_ok = False
                            confirmacion = True
                        else:
                            self.informar_mensaje(
                                f"Estadia de {estadia} minutos confirmada!"
                            )
                    except ConfirmacionProcesamientoException:
                        self.informar_mensaje(
                            "No se pudo procesar la confirmacion,"
                            + " intente nuevamente"
                        )
            except EstadiaProcesamientoException:
                self.informar_mensaje(
                    "No se pudo procesar la estadia, intente nuevamente"
                )
        return estadia

    def adquirir_techado(self) -> bool:
        techado = True
        while True:
            try:
                techado = self.obtener_confirmacion(
                    "¿Prefiere que su estacionamiento sea techado?"
                )
                break
            except ConfirmacionProcesamientoException:
                self.informar_mensaje(
                    "No se pudo procesar la respuesta, intente nuevamente"
                )
        return techado
