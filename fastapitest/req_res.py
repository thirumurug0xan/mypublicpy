from fastapi import FastAPI, Request, Response
from pydantic import BaseModel

app = FastAPI()

class UserLogin(BaseModel):
    email: str
    password: str

class UserPublic(BaseModel):
    email: str

@app.post("/login") #, response_model=UserPublic)
def login(data: UserLogin, response: Response, request: Request):
    # Parse request body into UserLogin
    # Manipulate response (e.g., set cookie)
    response.set_cookie(key="session", value="abc123", httponly=True)
    print(response)
    return UserPublic(email=data.email)

