# Premisa

Se requiere implementar un modulo de procesamiento de lenguaje natural para la gestion del ingreso de autos a un estacionamiento.

## Supuestos

- El estacionamiento está ubicado en Argentina.
- Se requiere pedir la patente, un estimativo de el tiempo de estadía y si se prefiere estacionamiento techado.

# Esta entrega

El sistema ya está implementado utilizando el modelo de OpenAI, whisper. 
Se refinará esta implementación utilizando un data set de audios de Argentina, para mejorar el rendimiento con el acento local.
El data set fué modificado para que las transcripciones contengan los numeros expesados en formato numerico y no como palabras (ej: 12 en lugar de doce). Esto es preferible para el procesamiento de patentes y horarios.

![Arquitectura whisper](whisper_architecture.svg)

En la imgen se muestra la arquitectura que whisper utiliza. Se reentrarán los parametros de la rnn qeu autoregresivamente predice los tokens, pero el mecanismo para embeber los tokens y la cnn utilizanda en la extraccíon de caracteristicas permanece intacta.

# El dataset

El dataset original está repartido en 3 archivos .zip, es_ar_female.zip, es_ar_male.zip y es_weather_messages.zip.
Estos contienen cada uno, una carpeta con multiples archivos .wav y un archivo .tsv.
Cada .tsv contiene 2 columnas donde la primera indica el nombre del archivo .wav al que se corresponde y la segunda la transcripcion del audio.
Todos los tsv fueron editados, como se expilcó anteriormente, para contener todos los numeros en formato numerico. La version editada tiene el sufijo '_numeros' añadido a su nombre.
Luego se los concatenó a todos en un solo .tsv llamado 'transcripciones.tsv'.

Por otro lado todos los .wav fueron movidos a una unica carpeta llamada 'audios'

# El codigo

## Comun

En el archivo 'comun.py' se definen todos los modulos comunes al entrenamiento y a la carga del modelo para probarlo.

- Toda la funcionalidad de entrenamiento se encapsula en la clase WhisperModelModule usando el LightningModule, esto solo organiza las funciones que regularmente se utilizan al enrenar una red neuronal un una clase. Aqui se crean los DataLoaders para entrenamiento y validacion, se configuran los optimizadores, y se definen los pasos de entrenamiento y validación.
- El optimizador que se utiliza es el AdamW, un tipo de decenso de gradiente estocastico basado en la estimación de los momentos de primer y segundo orden con un mecanismo de decaeimiento de los pesos. Se lo eligió en base a las recomendaciones de OpenAI sobre refinar su modelo. AdamW es recomendadopor su eficiencia tanto computacional como en el uso de memoria, muy util para modelos con muchos parametros como este, y por la invariancia ante el escalado diagonal de los gradientes.
- El dataset es encapsulado en la clase SpeechDataset
- La clase WhisperDataCollatorWithPadding es usada coma la collate_fn de los DataLoaders de pytorch. Se la encapsula en una clase por consistencia. Este se encarga de utilizar padding al final de cada transcripción para que todos los labels tengan el mismo largo.
- La clase Config define todos los metaparametros, que en este caso estan tomados segun las recomendaciones de OpenAI.

## Fine Tune

En el archivo 'fine_tune.py' se inicializan los modulos comunes y se ejecutan 10 epochs de entrenamiento. En cada epoch se guarda un checkpoint, en la carpeta artifacts, y se loggean todos los resultados del entrenamiento para ser vistos con tensorboard.

Para verlos se puede usar el comando siguiente

```shell
tensorboard  --logdir=tensor_logs/
```

## Ref Test



