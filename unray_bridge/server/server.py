import ray
from ray import rllib
from ray.rllib.agents.ppo import PPOTrainer

from fastapi import FastAPI
from pydantic import BaseModel

# Iniciar Ray
ray.init(ignore_reinit_error=True)

# Entrenador de PPO para el entorno CartPole
trainer = PPOTrainer(env="CartPole-v1")



# Definir el modelo de datos de entrada y salida para la API
class ActionRequest(BaseModel):
    state: list


class ActionResponse(BaseModel):
    action: int
    info: dict


# Crear la aplicación FastAPI
app = FastAPI()


# Endpoint para obtener la acción del agente
@app.post("/get_action", response_model=ActionResponse)
async def get_action(request: ActionRequest):
    # Obtener el estado del cuerpo de la solicitud
    state = request.state

    # Ejecutar la inferencia con RLlib: calcular la acción del agente
    action = trainer.compute_action(state)

    # Crear una respuesta con la acción y la información del agente
    response = ActionResponse(action=action, info={"message": "Action computed successfully"})

    return response


# Endpoint para obtener el estado del servidor
@app.get("/status")
async def status():
    return {"status": "Server is running"}
