from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from fastapi import FastAPI
import uvicorn
import database, model
# from routers import auth, user
import login, login, info, image

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model.Base.metadata.create_all(bind=database.engine)

# app.include_router(auth.router, prefix="/auth", tags=['?���?'])
# app.include_router(auth.router, prefix="/user", tags=['?��???'])
app.include_router(login.router, prefix="/login", tags=["login"])
app.include_router(info.router, prefix="/info", tags=["info"])
app.include_router(image.router, prefix="/image", tags=["image"])


@app.get("/")
async def root():
    return { "message" : "Hello World" }

# 127.0.0.1
# 192.168.0.8,
if __name__=="__main__":
    uvicorn.run(app, host="192.168.0.8", port=11112)

