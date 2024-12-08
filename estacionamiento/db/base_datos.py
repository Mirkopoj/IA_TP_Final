import random
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class Tiempo:
    fecha: datetime
    estadia: int = 60


@dataclass
class Reserva:
    patente: str
    tiempo: Tiempo
    posicion: int
    techado: bool


class BaseDatos:
    """
    Esta clase es una abstracción del accesso a la base de datos
    """

    def __init__(self):
        pass

    def tiene_reserva(self, patente: str) -> Optional[Reserva]:
        # Simula la consulta
        techado = random.choice([True, False])
        pos = 10 if techado else 1
        random.choice(
            [
                Reserva(patente, Tiempo(fecha=datetime.now()), pos, techado),
                None,
            ]
        )

    def verificar_espacios(
        self, patente: str, tiempo: Tiempo, techado: bool
    ) -> Optional[Reserva]:
        # Simula la consulta, si hay espacio le asigna uno y retorna la reserva
        # Si no no retorna nada
        pos = 10 if techado else 1
        return random.choice([Reserva(patente, tiempo, pos, techado), None])

    def registrar_vehiculo(self, reserva: Reserva):
        # Simula el registro
        print(
            f"Vehiculo con {reserva.patente} registrado en la posicion {reserva.posicion} durante {reserva.tiempo.estadia} minutos."
        )
