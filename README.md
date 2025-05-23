 [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) <br>
 ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
 ![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
 
 La presente es una red neuronal densa (completamente conectada), que predice el crecimiento de plantas basado en 7 de sus aspectos: Soil_Type, Sunlight_Hours, Water_Frequency, Fertilizer_Type, Temperature, Humidity, Growth_Milestone (La medida objetivo).
Fue diseñada para problemas de clasificación binaria.
Tiene 3 capas:
La capa de entrada, que acepta los 14-16 rasgos generados por el preprocesamiento 
Las capas ocultas, que son 2:
La primera capa con 64 neuronas y función de activación ReLU
La segunda capa con 32 neuronas, también con ReLU
Ambas capas tienen un dropout del 30% en cada capa para prevenir overfitting
La capa de salida, con 1 neurona con activación sigmoid para probabilidad binaria (0-1)

**`Detalles de la implementación y los parámetros utilizados.`**
-------------------
Hubieron 2 tipos de variable:
Variables numéricas (Sunlight_Hours, Temperature, Humidity): Estandarizadas (media=0, desviación=1)
Variables categóricas (Soil_Type, Water_Frequency, Fertilizer_Type): One-Hot Encoding
Se observó que el modelo alcanzó ~70% de certidumbre en entrenamiento pero ~56% en pruebas.
La perdida de validación aumentó (0.69 a 0.84) mientras que el de entrenamiento disminuyó (0.74 a 0.55)
Se usaron 50 epochs.
**`Resultados y posibles mejoras`**
-------------------
La certeza en pruebas finales fue: 56.4%, que es ligeramente mejor que el azar, 50%. Esto demuestra que el modelo si aprende, aunque es posible que el tamaño limitado del dataset (200 muestras).
Uno de los problemas principales observados es el overfitting; el modelo memoriza entrenamiento pero no generaliza. Una posible causa es el desbalance de clases. Se debería considerar que growth_Milestone sea más descriptivo.

Hay muchas mejoras posibles a hacer para el modelo, incluyendo pero no limitándose a:
Early Stopping, deteniendo el entrenamiento cuando una métrica deja de mejorar.
Regularización, con la función kernel_regularizer de keras.
Balanceo de clases, ajustando el peso para hallar resultados más relevantes.
Feature engineering, la creación de rasgos interactivos, (Sunlight*Temperature)
El uso de otras arquitecturas, que sean más adaptadas a modelos más pequeños
**`Conclusiones`**
-------------------
Este análisis sugiere que aunque el modelo funciona, necesita ajustes para mejorar su capacidad de generalización. El enfoque debería estar en reducir el overfitting y posiblemente recolectar más datos si es posible.

"modelo.py" corresponde al codigo para crear y entrenar el modelo, deberia devolver una punteria de ~56%.
"plant_growth_data.csv" corresponde a los datos con los que se entrenan el modelo
"plant_growth_predictor.keras" es el modelo entrenado por nosotros, que se regeneraria al correr "modelo.py".