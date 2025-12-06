import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
import sys

# Configuración del modelo
BASE_MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-Instruct"
ADAPTER_MODEL_NAME = "IbarraOrtizDev/agatec_cafe"

SYSTEM_PROMPT = """Eres un asistente experto en café colombiano entrenado con información de Cenicafé.
Tu objetivo es responder preguntas sobre cultivo, variedades, productividad y prácticas agronómicas del café en Colombia.
Proporciona respuestas precisas, informativas y basadas en conocimiento técnico."""


def check_gpu():
    """Verifica si hay GPU disponible"""
    if torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"GPU detectada: {gpu_name}")
            print(f"   Memoria VRAM: {gpu_memory:.2f} GB")
            return True
        except Exception as e:
            print(f"Error al obtener información de GPU: {e}")
            return True  # Asumir que hay GPU pero sin detalles
    else:
        print("No se detectó GPU")
        print("Este modelo requiere GPU para funcionar")
        return False


def load_model():
    """Carga el modelo base y el adaptador fine-tuned"""
    
    # Verificar GPU
    has_gpu = check_gpu()
    if not has_gpu:
        error_msg = (
            "Este Space requiere GPU para funcionar correctamente.\n\n"
        )
        raise RuntimeError(error_msg)
    
    print("\n" + "="*50)
    print("CARGANDO MODELO DE CAFÉ COLOMBIANO")
    print("="*50)
    
    print("\nPaso 1/3: Cargando modelo base...")
    print(f"   Modelo: {BASE_MODEL_NAME}")
    
    # Determinar el dtype más apropiado
    if torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
        print("   Precisión: bfloat16 (óptimo)")
    else:
        dtype = torch.float16
        print("   Precisión: float16")
    
    # Crear directorio de offload si no existe
    import os
    os.makedirs("offload", exist_ok=True)
    
    try:
        # Cargar modelo base con optimizaciones agresivas para T4
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            torch_dtype=dtype,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            offload_folder="offload",
            offload_state_dict=True,
            max_memory={0: "14GB", "cpu": "30GB"}  # Reservar memoria para offload
        )
        print("Modelo base cargado")
        
        print("\nPaso 2/3: Cargando tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL_NAME,
            trust_remote_code=True
        )
        tokenizer.pad_token = tokenizer.eos_token
        print("Tokenizer cargado")
        
        print("\nPaso 3/3: Cargando adaptador fine-tuned...")
        print(f"Adaptador: {ADAPTER_MODEL_NAME}")
        model = PeftModel.from_pretrained(base_model, ADAPTER_MODEL_NAME)
        model.eval()
        print("Adaptador cargado")
        
        print("\n" + "="*50)
        print("MODELO LISTO PARA USAR")
        print("="*50 + "\n")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"\nERROR AL CARGAR EL MODELO:")
        print(f"   {str(e)}")
        raise


def generate_response_clean(model, tokenizer, question, max_tokens=300):
    """Genera una respuesta limpia del modelo"""
    
    # Construir el prompt en formato Llama 3
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{SYSTEM_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>
{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
    
    # Tokenizar entrada
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generar respuesta
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decodificar respuesta completa
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # Extraer SOLO la respuesta del asistente
    if "<|start_header_id|>assistant<|end_header_id|>" in full_response:
        parts = full_response.split("<|start_header_id|>assistant<|end_header_id|>")
        response = parts[-1].strip()
    else:
        response = full_response
    
    # Limpiar tokens especiales residuales
    response = response.replace("<|eot_id|>", "").strip()
    response = response.replace("<|end_of_text|>", "").strip()
    response = response.replace("<|begin_of_text|>", "").strip()
    
    # Limpiar headers que puedan haber quedado
    lines = response.split('\n')
    clean_lines = []
    
    for line in lines:
        line_lower = line.lower().strip()
        # Saltar líneas que sean solo headers
        if line_lower in ['system', 'user', 'assistant']:
            continue
        # Saltar líneas que contengan el prompt del sistema
        if 'eres un asistente' in line_lower:
            continue
        clean_lines.append(line)
    
    response = '\n'.join(clean_lines).strip()
    
    return response
# ============================================
print("\nINICIALIZANDO APLICACIÓN")
print(f"   Python: {sys.version.split()[0]}")
print(f"   PyTorch: {torch.__version__}")
print(f"   CUDA disponible: {torch.cuda.is_available()}")

try:
    model, tokenizer = load_model()
    MODEL_LOADED = True
    print("Aplicación lista para recibir consultas\n")
    
except Exception as e:
    print(f"\nERROR FATAL: No se pudo cargar el modelo")
    print(f"   Detalle: {e}\n")
    MODEL_LOADED = False
    model = None
    tokenizer = None


def chat_response(message, history):
    """Función principal para la interfaz de Gradio"""
    
    if not MODEL_LOADED:
        return (
            "**ERROR: Modelo no cargado**\n\n"
        )
    
    if not message or message.strip() == "":
        return "Por favor, escribe una pregunta sobre café colombiano."
    
    try:
        response = generate_response_clean(model, tokenizer, message)
        return response
    except Exception as e:
        return f"Error al generar respuesta: {str(e)}\n\nPor favor, intenta de nuevo o con una pregunta diferente."


examples = [
    ["¿Cuáles son las variedades de café resistentes a la roya?"],
    ["¿Cuál es la productividad promedio de café en Colombia?"],
    ["¿Qué diferencias hay entre Cenicafé 1 y Castillo?"],
    ["¿Cuándo debo sembrar café en Antioquia?"],
    ["¿Cuáles son las 8 prácticas agronómicas fundamentales?"],
    ["¿Cuánto ahorra Colombia por usar variedades resistentes a roya?"],
    ["¿Qué cuidados necesita un cafetal joven?"],
    ["¿Cómo prevenir la broca del café?"]
]

# Mensaje de estado dinámico
if MODEL_LOADED:
    status_message = "**Sistema listo** - Modelo cargado correctamente"
else:
    status_message = "**GPU requerida** - Configura hardware en Settings"

# Descripción de la interfaz
description = f"""
**Asistente experto en café colombiano** entrenado con información de Cenicafé.
{status_message}
### 💬 Pregúntame sobre:
- 🌱 Variedades de café y resistencia a enfermedades
- 🌾 Prácticas agronómicas y manejo del cultivo
- 📊 Productividad y economía cafetera
- 📅 Épocas de siembra según la región
- 🛡️ Control de plagas y enfermedades
- 🌡️ Condiciones climáticas óptimas
### 🤖 Información técnica:
- **Modelo base:** Meta-Llama-3.1-8B-Instruct
- **Entrenamiento:** Fine-tuned con datos de Cenicafé
- **Método:** PEFT (Parameter-Efficient Fine-Tuning)
"""

# Crear interfaz de chat
demo = gr.ChatInterface(
    fn=chat_response,
    title="☕ Chatbot de Café Colombiano",
    description=description,
    examples=examples if MODEL_LOADED else None,
)

# ============================================
# LANZAR APLICACIÓN
# ============================================

if __name__ == "__main__":
    demo.launch()
