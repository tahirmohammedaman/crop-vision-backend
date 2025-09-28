from pydantic import BaseModel


class UserBase(BaseModel):
    username: str


class UserLogin(UserBase):
    password: str


class UserOut(UserBase):
    id: int
