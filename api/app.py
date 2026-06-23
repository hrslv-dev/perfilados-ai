from fastapi import FastAPI

from api.routes import router

# O que aconteceu aqui?
app = FastAPI()

app.include_router(router)
