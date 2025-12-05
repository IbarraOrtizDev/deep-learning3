# AGATEC CAFÉ

**Agente Generativo para Asistencia Técnica en el Cultivo del Café**

## Descripción del Proyecto

AGATEC CAFÉ es un chatbot especializado basado en inteligencia artificial diseñado para democratizar el acceso al conocimiento técnico sobre el cultivo del café en Colombia. El sistema proporciona asistencia técnica rápida y confiable basada en las directrices científicas de Cenicafé (Centro Nacional de Investigaciones de Café).

### Propósito

- **Transferencia de Conocimiento**: Transformar la investigación académica de Cenicafé en consejos accesibles
- **Democratización**: Herramienta gratuita para pequeños y medianos caficultores
- **Soporte Técnico**: Respuestas rápidas a preguntas sobre cultivo
- **Sostenibilidad**: Promover las mejores prácticas agrícolas

### Características Principales

- Asistencia técnica especializada en café colombiano
- Conocimiento basado en 508 preguntas y respuestas de Cenicafé
- Interfaz web intuitiva mediante Gradio
- Modelo fine-tuned Llama 3.1 8B con LoRA
- Optimización de memoria con cuantización 4-bit
- Parámetros configurables (temperatura, tokens, top-p)

### Temas que Cubre el Chatbot

- Variedades de café (Colombia, Castillo, Cenicafé 1, Tabi, etc.)
- Manejo de enfermedades (especialmente roya del café)
- Prácticas agronómicas fundamentales
- Productividad y economía cafetera
- Épocas de siembra por región
- Manejo de suelos y nutrición
- Producción de semillas y plántulas

## Tecnologías Utilizadas

### Modelo de IA

- **Base**: Meta Llama 3.1 8B Instruct (versión optimizada por Unsloth)
- **Fine-tuning**: LoRA (Low-Rank Adaptation) - Adaptadores especializados
- **Cuantización**: 4-bit (bitsandbytes) para eficiencia de memoria
- **Adaptador**: IbarraOrtizDev/agatec_cafe (hospedado en HuggingFace)

### Stack Tecnológico

- **Python 3.8+**
- **PyTorch**: Framework de deep learning
- **Transformers** (Hugging Face): Biblioteca de modelos de lenguaje
- **PEFT**: Parameter-Efficient Fine-Tuning (LoRA)
- **Gradio**: Framework para interfaz web interactiva
- **Accelerate**: Optimización de inferencia

### Dataset

- **Fuente**: Cenicafé (Centro Nacional de Investigaciones de Café)
- **Registros**: 508 pares pregunta-respuesta
- **Categorías**: 15 categorías principales
- **Subcategorías**: 192 subcategorías
- **Archivo**: `dataset/dt_cafe.csv`

## Manual de Instalación

### Requisitos del Sistema

#### Requisitos Mínimos

- **Sistema Operativo**: Linux, Windows, macOS
- **Python**: 3.8 o superior
- **RAM**: 16 GB (mínimo)
- **Almacenamiento**: 20 GB libres

#### Requisitos Recomendados

- **GPU**: NVIDIA con 8+ GB VRAM (CUDA compatible)
- **RAM**: 32 GB
- **Almacenamiento**: SSD con 30 GB libres

**Nota**: El modelo puede ejecutarse en CPU, pero la inferencia será más lenta.

### Paso 1: Clonar el Repositorio

```bash
git clone https://github.com/IbarraOrtizDev/deep-learning3
cd deep-learning3
```

### Paso 2: Crear Entorno Virtual (Recomendado)

```bash
# Crear entorno virtual
python -m venv venv

# Activar entorno virtual
# En Linux/macOS:
source venv/bin/activate

# En Windows:
venv\Scripts\activate
```

### Paso 3: Instalar Dependencias

```bash
pip install -r requirements.txt
```

#### Contenido de requirements.txt

- `gradio>=4.0.0` - Framework de interfaz web
- `transformers>=4.51.3` - Biblioteca de modelos Hugging Face
- `torch>=2.0.0` - PyTorch framework
- `peft>=0.18.0` - Fine-tuning eficiente (LoRA)
- `accelerate>=0.20.0` - Optimización de inferencia
- `safetensors>=0.4.3` - Serialización segura de tensores
- `bitsandbytes>=0.41.0` - Cuantización 4-bit
- `sentencepiece>=0.1.99` - Tokenizador para Llama

### Paso 4: Verificar Instalación de CUDA (Opcional - GPU)

Si tienes GPU NVIDIA, verifica la instalación de CUDA:

```bash
python -c "import torch; print(f'CUDA disponible: {torch.cuda.is_available()}')"
```

### Paso 5: Configuración Adicional (Opcional)

El modelo se descarga automáticamente desde HuggingFace en la primera ejecución. Asegúrate de tener conexión a internet estable para la descarga inicial (~16 GB).

## Guía de Usuario

### Ejecución de la Aplicación

#### Iniciar el Chatbot

```bash
python app.py
```

#### Salida Esperada

```
GPU disponible: NVIDIA GeForce RTX 3090
Memoria GPU: 24.00 GB
Cargando modelo base...
Modelo cargado exitosamente!

Lanzando aplicación...
Running on local URL:  http://0.0.0.0:7860
```

#### Acceder a la Interfaz Web

1. Abre tu navegador web
2. Visita: `http://localhost:7860`
3. La interfaz del chatbot estará lista para usar

### Uso del Chatbot

#### Interfaz de Usuario

La aplicación muestra:

- **Título**: "Chatbot de Café Colombiano"
- **Descripción**: Información sobre las capacidades del asistente
- **Ejemplos**: Preguntas pre-configuradas para empezar
- **Área de chat**: Para escribir tus preguntas
- **Controles avanzados**: Sliders para ajustar parámetros

#### Ejemplos de Preguntas

Haz clic en cualquiera de estos ejemplos o escribe tu propia pregunta:

1. "¿Cuáles son las variedades de café resistentes a la roya?"
2. "¿Cuál es la productividad promedio de café en Colombia?"
3. "¿Qué diferencias hay entre Cenicafé 1 y Castillo?"
4. "¿Cuándo debo sembrar café en Antioquia?"
5. "¿Cuáles son las 8 prácticas agronómicas fundamentales?"
6. "¿Cuánto ahorra Colombia por usar variedades resistentes a roya?"

#### Parámetros Configurables

**1. Máximo de Tokens** (128-1024)
- **Valor por defecto**: 512
- **Descripción**: Controla la longitud máxima de la respuesta
- **Uso**: Aumenta para respuestas más detalladas, reduce para respuestas concisas

**2. Temperature** (0.1-1.5)
- **Valor por defecto**: 0.7
- **Descripción**: Controla la creatividad del modelo
- **Bajo (0.1-0.5)**: Respuestas más conservadoras y precisas
- **Medio (0.6-0.8)**: Balance entre precisión y variedad
- **Alto (0.9-1.5)**: Respuestas más creativas y diversas

**3. Top-p** (0.1-1.0)
- **Valor por defecto**: 0.9
- **Descripción**: Nucleus sampling para diversidad
- **Bajo**: Respuestas más enfocadas
- **Alto**: Mayor variedad en las respuestas

#### Mejores Prácticas

1. **Formula preguntas claras y específicas** sobre café colombiano
2. **Usa los ejemplos** como guía para el formato de preguntas
3. **Ajusta los parámetros** según tus necesidades:
   - Para respuestas técnicas precisas: temperature=0.3, top_p=0.7
   - Para respuestas balanceadas: valores por defecto
   - Para explorar alternativas: temperature=1.0, top_p=0.95
4. **Ten paciencia**: La primera respuesta puede tomar 10-30 segundos mientras el modelo se carga

### Acceso desde Otros Dispositivos

La aplicación se ejecuta en `0.0.0.0:7860`, permitiendo acceso desde otros dispositivos en la misma red:

```
http://<TU_IP_LOCAL>:7860
```

Para obtener un enlace público temporal, edita `app.py` línea 193:

```python
demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=True,  # Cambiar de False a True
    show_error=True
)
```

### Detener la Aplicación

Presiona `Ctrl+C` en la terminal donde se ejecuta la aplicación.

## Estructura del Proyecto

```
deep-learning3/
├── app.py                          # Aplicación principal Gradio
├── requirements.txt                # Dependencias Python
├── dataset/
│   └── dt_cafe.csv                 # Dataset de entrenamiento (508 registros)
├── documentacion/
│   └── Ibarra_Ortiz_Edwin_Alexander_EA3_GenerativeAI.pdf
├── Ibarra_Ortiz_Edwin_Alexander_EA3_GenerativeAI_Notebook.ipynb
├── fix_data.js                     # Utilidad de limpieza de datos
└── README.md                       # Este archivo
```

## Resultados del Modelo

### Métricas de Evaluación

- **Training Loss**: 2.7035 → 0.6169 (77.2% reducción)
- **Validation Loss**: 0.7066
- **Train-Eval Gap**: 0.0897 (sin overfitting)
- **BLEU Score**: 3.4024
- **ROUGE-1**: 0.2354
- **ROUGE-2**: 0.0529
- **ROUGE-L**: 0.1787

### Configuración de Fine-Tuning

- **LoRA**: r=16, alpha=16, dropout=0.05
- **Learning Rate**: 5e-5
- **Épocas**: 3
- **Batch Size**: 2
- **Gradient Accumulation**: 4
- **Max Sequence Length**: 2048

## Solución de Problemas

### Error: "CUDA out of memory"

**Solución**:
- Reduce `max_tokens` a 256-384
- Cierra otras aplicaciones que usen GPU
- Usa CPU (automático si no hay GPU disponible)

### Error: "Model not loaded"

**Solución**:
- Verifica conexión a internet (primera descarga)
- Asegúrate de tener suficiente espacio en disco (20+ GB)
- Revisa que las dependencias estén instaladas correctamente

### Respuestas Lentas

**Causas comunes**:
- Ejecución en CPU (normal, espera 30-60 segundos)
- Primera inferencia (carga de modelo, espera 10-30 segundos)
- `max_tokens` muy alto

**Solución**:
- Usa GPU si está disponible
- Reduce `max_tokens` a 256-512
- Ten paciencia en la primera consulta

### Error de Instalación de `bitsandbytes`

**En Windows**:
```bash
pip install bitsandbytes-windows
```

**En Linux**:
```bash
pip install bitsandbytes
```

## Contribuciones

Este proyecto es parte de una investigación académica sobre IA generativa aplicada al sector agrícola colombiano.

## Licencia

Este proyecto está desarrollado con fines educativos y de investigación.

## Contacto

**Autor**: Edwin Alexander Ibarra Ortiz
**Institución**: IUD (Institución Universitaria Digital)
**Curso**: Deep Learning - EA3 Generative AI

## Referencias

- **Cenicafé**: [Centro Nacional de Investigaciones de Café de Colombia](https://cenicafe.org/)
- **Meta Llama**: https://ai.meta.com/llama/
- **Unsloth**: https://github.com/unslothai/unsloth
- **Hugging Face**: https://huggingface.co/
- **Gradio**: https://www.gradio.app/