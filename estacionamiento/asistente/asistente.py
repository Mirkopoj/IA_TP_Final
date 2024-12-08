import pyttsx3


class Asistente:
    def __init__(self, velocidad: int = 150, volumen: float = 1.0):

        self.motor = pyttsx3.init()

        self.motor.setProperty("rate", velocidad)
        self.motor.setProperty("volume", volumen)

        voces = self.motor.getProperty("voices")
        voz_español = [v for v in voces if "es" in v.languages][0]
        self.motor.setProperty("voice", voz_español.id)

    def habla(self, mensaje: str):
        self.motor.say(mensaje)
        self.motor.runAndWait()
