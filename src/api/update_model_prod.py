import mlflow

from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("file:../notebooks/mlruns")
client = MlflowClient()

client.transition_model_version_stage(
    name="CreditRiskModel",
    version=1,  # replace with your version number
    stage="Production",
)
