from fastapi import FastAPI

app = FastAPI()

@app.get("/usarios")
def obtener_usuarios():
    return [
        {"nombre": "Rosa", "id": 1},
        {"nombre": "Juan", "id": 2}
    ]