import math
import os
import re
import wave
from typing import Optional

import pyaudio
import torch
import whisper


class ProcesamientoException(Exception):
    pass


class PatenteProcesamientoException(ProcesamientoException):
    pass


class EstadiaProcesamientoException(ProcesamientoException):
    pass


class ConfirmacionProcesamientoException(ProcesamientoException):
    pass


class IA:
    FORMATO = pyaudio.paInt16
    CANALES = 1
    RATE = 44100
    CHUNK = 512
    SEGUNDOS_GRABACION = 7
    NOMBRE_ARCHIVO = "audiotemp.wav"
    AFIRMATIVO = [
        "bueno",
        "joya",
        "dale",
        "si",
        "sí",
        "perfecto",
        "buenisimo",
        "buenisímo",
    ]
    NEGATIVO = [
        "no",
        "nah",
        "neh",
        "ni",
    ]

    def __init__(self, audio: pyaudio.PyAudio, id_dispositivo: int):
        self.audio = audio
        dispositivo = "cpu"
        if torch.cuda.is_available():
            dispositivo = "cuda"
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        self.modelo = whisper.load_model("small", device=dispositivo)
        for p in self.modelo.encoder.parameters():
            p.requires_grad = False
        checkpoint_path = "artifacts/checkpoint/checkpoint-epoch=0009.ckpt"
        state_dict = torch.load(checkpoint_path)['state_dict']
        self.modelo.load_state_dict(state_dict, strict=False)
        self.id_dispositivo = id_dispositivo
        pass

    def recibir_audio_a_str(self) -> str:
        stream = self.audio.open(
            format=self.FORMATO,
            channels=self.CANALES,
            rate=self.RATE,
            input=True,
            input_device_index=self.id_dispositivo,
            frames_per_buffer=self.CHUNK,
        )

        print("Escuchando...")

        frames_grabados = []
        for _a in range(0, math.ceil(self.RATE / self.CHUNK * self.SEGUNDOS_GRABACION)):
            data = stream.read(self.CHUNK)
            frames_grabados.append(data)

        stream.stop_stream()
        stream.close()

        archivo = wave.open(self.NOMBRE_ARCHIVO, "wb")
        archivo.setnchannels(self.CANALES)
        archivo.setsampwidth(self.audio.get_sample_size(self.FORMATO))
        archivo.setframerate(self.RATE)
        archivo.writeframes(b"".join(frames_grabados))
        archivo.close()

        resultado = self.modelo.transcribe(self.NOMBRE_ARCHIVO, language="es")

        resultado_str = resultado["text"].__str__().lower().strip()
        resultado_str = "".join(
            c for c in resultado_str if c.isalnum() or c.isspace())
        print(f"Se entendio {resultado_str}")
        return resultado_str

    def procesar_patente(self) -> str:
        texto_desde_audio = self.recibir_audio_a_str()

        valido, patente = self.extraer_patente(texto_desde_audio)

        if not valido:
            raise PatenteProcesamientoException()

        return patente

    def isdig(self, car: str) -> bool:
        return car.isdigit()

    def extraer_tiempo(self, texto: str):
        texto = texto.replace("una", "1")
        texto = texto.replace("uno", "1")
        texto = texto.replace("dos", "2")
        texto = texto.replace("tres", "3")
        texto = texto.replace("cuatro", "4")
        texto = texto.replace("cinco", "5")
        texto = texto.replace("seis", "6")
        texto = texto.replace("siete", "7")
        texto = texto.replace("ocho", "8")
        texto = texto.replace("nueve", "9")
        texto = texto.replace("media", "30")
        texto = texto.replace("cuarto", "15")
        palabras = texto.strip().split()

        numeros = []
        for palabra in palabras:
            if palabra[0].isdigit():
                num = "".join(filter(self.isdig, palabra))
                numeros.append(int(num))
        if len(numeros) < 1:
            numeros.append(0)
        if len(numeros) < 2:
            numeros.append(0)
        return (numeros[0], numeros[1])

    def procesar_estadia(self) -> int:
        texto_desde_audio = self.recibir_audio_a_str()

        horas, minutos = self.extraer_tiempo(texto_desde_audio)

        minutos_total = (horas * 60) + minutos

        if minutos_total <= 0:
            raise EstadiaProcesamientoException()

        return minutos_total

    def procesar_confirmacion(self) -> bool:
        texto_desde_audio = self.recibir_audio_a_str()

        confirmacion = self.extraer_confirmacion(texto_desde_audio)

        if confirmacion is None:
            raise ConfirmacionProcesamientoException()

        return confirmacion

    def extraer_confirmacion(self, texto: str) -> Optional[bool]:
        for palabra in texto.split():
            if palabra in self.AFIRMATIVO:
                return True
            if palabra in self.NEGATIVO:
                return False
        return None

    def extraer_patente(self, texto: str):
        texto = texto.upper()
        texto_limpio = "".join(c for c in texto if c.isalnum())

        patrones = {
            "actual": r"[A-Z]{2}\d{3}[A-Z]{2}",  # AA123BB
            "anterior": r"[A-Z]{3}\d{3}",  # ABC123
            "moto": r"[A-Z]\d{3}[A-Z]{3}",  # A123BCD
        }

        for _, patron in patrones.items():
            match = re.search(patron, texto_limpio)
            if match:
                return (True, match.group())

        return (False, texto_limpio)
