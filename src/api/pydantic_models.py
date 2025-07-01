from pydantic import BaseModel


class PredictionRequest(BaseModel):
    num__Amount: float
    num__Value: float
    num__CustomerId_TransactionCount: float
    num__CustomerId_TotalAmount: float
    num__CustomerId_AvgAmount: float
    num__CustomerId_AmountStd: float
    num__CustomerId_AmountSkew: float
    num__ProductCategory_TransactionCount: float
    num__ProductCategory_TotalAmount: float
    num__ProductCategory_AvgAmount: float
    num__ProductCategory_AmountStd: float
    num__ProductCategory_AmountSkew: float
    num__TransactionHour: float
    num__TransactionDay: float
    num__TransactionDayOfWeek: float
    num__TransactionMonth: float
    num__TransactionYear: float
    num__IsWeekend: float


class PredictionResponse(BaseModel):
    risk_probability: float
