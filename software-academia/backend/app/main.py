from fastapi import FastAPI

app = FastAPI(title="Software Academia API")

@app.get("/")
def root():
    return {"message": "Software Academia API is running"}
