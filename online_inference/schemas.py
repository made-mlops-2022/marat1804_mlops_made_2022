from typing import Literal

from pydantic import BaseModel, validator


class InputData(BaseModel):
    age: float
    sex: Literal[0, 1]
    cp: Literal[0, 1, 2, 3]
    trestbps: float
    chol: float
    fbs: Literal[0, 1]
    restecg: Literal[0, 1, 2]
    thalach: float
    exang: Literal[0, 1]
    oldpeak: float
    slope: Literal[0, 1, 2]
    ca: Literal[0, 1, 2, 3]
    thal: Literal[0, 1, 2]

    @validator('age')
    def age_validator(cls, value):
        if value <= 0 or value > 100:
            raise ValueError('Age must be between 1 and 100')
        return value

    @validator('chol')
    def chol_validator(cls, value):
        if value <= 0 or value > 1000:
            raise ValueError('Chol must be between 1 and 1000')
        return value

    @validator('trestbps')
    def trestbps_validator(cls, value):
        if value <= 0 or value > 300:
            raise ValueError('Trestbps must be between 1 and 300')
        return value

    @validator('thalach')
    def thalach_validator(cls, value):
        if value <= 0 or value > 300:
            raise ValueError('Thalach must be between 1 and 300')
        return value

    @validator('oldpeak')
    def oldpeak_validator(cls, value):
        if value <= 0 or value > 10:
            raise ValueError('Oldpeak must be between 1 and 10')
        return value
