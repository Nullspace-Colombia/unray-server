from fastapi import FastAPI
from fastapi.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware

app = FastAPI()

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite todas las fuentes (para desarrollo, ajusta esto en producción)
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos los métodos HTTP (GET, POST, etc.)
    allow_headers=["*"],  # Permite todos los encabezados
)

@app.get("/status")
async def get_status():
    # Si el servidor está funcionando correctamente, devuelve un estado 200
    return JSONResponse(content={"status": "online"}, status_code=200)

# Si quieres manejar un caso de error o desconexión
@app.get("/status/error")
async def get_status_error():
    # Simula un error o desconexión
    return JSONResponse(content={"status": "offline"}, status_code=503)
