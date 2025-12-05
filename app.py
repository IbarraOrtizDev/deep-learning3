import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import warnings
import gc
warnings.filterwarnings('ignore')

# Configuración del modelo
MODEL_BASE = "unsloth/meta-llama-3.1-8b-instruct-unsloth-bnb-4bit"
ADAPTER_PATH = "IbarraOrtizDev/agatec_cafe"

# Variables globales para el modelo y tokenizador
model = None
tokenizer = None

def load_model():
    global model, tokenizer

    # Limpiar memoria antes de cargar
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Cargando modelo base...")
    # Cargar el tokenizador
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_BASE,
        trust_remote_code=True
    )

    # Configurar según el dispositivo disponible
    if torch.cuda.is_available():
        # Con GPU, usar float16
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_BASE,
            dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map="auto"
        )
    else:
        # Con CPU, usar float16 para reducir memoria a la mitad
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_BASE,
            dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )

    # Cargar los adaptadores LoRA
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)

    # Fusionar adaptadores con el modelo base para ahorrar memoria
    model = model.merge_and_unload()
    model.eval()

    # Limpiar memoria después de cargar
    gc.collect()

    print("Modelo cargado exitosamente!")
    return model, tokenizer

def generate_response(message, history, max_tokens=512, temperature=0.7, top_p=0.9):
    global model, tokenizer

    if model is None or tokenizer is None:
        return "Error: El modelo no está cargado. Por favor, reinicie la aplicación."

    # Construir el prompt con formato de instrucción
    prompt = f"""### Instrucción:
Eres un asistente experto en café colombiano entrenado con información de Cenicafé.
Tu objetivo es responder preguntas sobre cultivo, variedades, productividad y prácticas agronómicas del café en Colombia.
Proporciona respuestas precisas, informativas y basadas en conocimiento técnico.

### Pregunta:
{message}

### Respuesta:
"""

    # Tokenizar el prompt
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    )

    # Mover a GPU si está disponible
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    # Generar respuesta
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.15,
        )

    # Decodificar la respuesta
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extraer solo la respuesta generada (después de "### Respuesta:")
    if "### Respuesta:" in full_response:
        response = full_response.split("### Respuesta:")[-1].strip()
    else:
        response = full_response[len(prompt):].strip()

    return response

def create_interface():
    # Cargar el modelo al iniciar
    load_model()

    # Ejemplos de preguntas
    examples = [
        ["¿Cuáles son las variedades de café resistentes a la roya?"],
        ["¿Cuál es la productividad promedio de café en Colombia?"],
        ["¿Qué diferencias hay entre Cenicafé 1 y Castillo?"],
        ["¿Cuándo debo sembrar café en Antioquia?"],
        ["¿Cuáles son las 8 prácticas agronómicas fundamentales?"],
        ["¿Cuánto ahorra Colombia por usar variedades resistentes a roya?"]
    ]

    # Crear interfaz de chat
    demo = gr.ChatInterface(
        fn=generate_response,
        title="☕ Chatbot de Café Colombiano",
        description="""
        **Asistente experto en café colombiano** entrenado con información de Cenicafé.

        Pregúntame sobre:
        - Variedades de café resistentes a la roya
        - Prácticas agronómicas
        - Productividad y economía cafetera
        - Épocas de siembra por región
        - Manejo de enfermedades

        **Modelo:** unsloth/meta-llama-3.1-8b-instruct-unsloth-bnb-4bit (Experimento 6)
        """,
        examples=examples,
        additional_inputs=[
            gr.Slider(
                minimum=128,
                maximum=1024,
                value=512,
                step=64,
                label="Máximo de tokens",
                info="Controla la longitud de la respuesta"
            ),
            gr.Slider(
                minimum=0.1,
                maximum=1.5,
                value=0.7,
                step=0.1,
                label="Temperature",
                info="Controla la creatividad (bajo=conservador, alto=creativo)"
            ),
            gr.Slider(
                minimum=0.1,
                maximum=1.0,
                value=0.9,
                step=0.05,
                label="Top-p",
                info="Nucleus sampling para diversidad"
            )
        ],
    )

    return demo

if __name__ == "__main__":
    # Verificar si hay GPU disponible
    if torch.cuda.is_available():
        print(f"GPU disponible: {torch.cuda.get_device_name(0)}")
        print(f"Memoria GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("No se detectó GPU. Usando CPU (puede ser más lento)")

    demo = create_interface()

    print("\nLanzando aplicación...")
    demo.launch(
        server_name="0.0.0.0",  # Permite acceso desde cualquier IP
        server_port=7860,        # Puerto por defecto de Gradio
        share=False,             # Cambia a True para obtener un link público temporal
        show_error=True
    )
