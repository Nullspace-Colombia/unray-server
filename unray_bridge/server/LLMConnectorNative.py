import torch
from transformers import pipeline

# ID del modelo que quieres usar
model_id = "meta-llama/Llama-3.2-3B-Instruct"

# Inicialización del pipeline de generación de texto
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Mensaje inicial del sistema para orientar al modelo
messages = [
    {"role": "system",
     "content": "You are an AI assistant who helps users with information about artificial intelligence, machine learning, and technology. Provide helpful, clear, and concise answers."}
]

# Bucle para interactuar continuamente con el modelo
while True:
    user_input = input("User: ")  # Solicitar entrada del usuario
    if user_input.lower() in ["exit", "quit", "bye"]:
        print("Exiting the chat.")
        break  # Salir del bucle si el usuario escribe "exit", "quit" o "bye"

    # Agregar el mensaje del usuario al contexto
    messages.append({"role": "user", "content": user_input})

    # Generar respuesta del modelo
    outputs = pipe(
        messages,
        max_new_tokens=256,  # Limita la cantidad de tokens generados
    )

    # Obtener la respuesta del modelo
    model_output = outputs[0]["generated_text"]

    # Mostrar solo la respuesta generada por el modelo
    print(f"AI Assistant: {model_output}")

    # Agregar la respuesta del modelo al contexto para la próxima iteración
    messages.append({"role": "assistant", "content": model_output})
