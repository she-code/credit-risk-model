import mlflow

# mlflow.set_tracking_uri("file:../notebooks/mlruns")
# client = mlflow.MlflowClient()

# # List all registered models
# models = client.list_registered_models()
# for m in models:
#     print(f"Name: {m.name}")
#     for mv in m.latest_versions:
#         print(f" - Version: {mv.version}, Stage: {mv.current_stage}, Status: {mv.status}")

from mlflow.tracking import MlflowClient
mlflow.set_tracking_uri("file:../notebooks/mlruns")
client = MlflowClient()

client.transition_model_version_stage(
    name="CreditRiskModel",
    version=1,          # replace with your version number
    stage="Production"
)

# models = client.search_registered_models()
# for model in models:
#     print(f"Model Name: {model.name}")
#     for version in model.latest_versions:
#         print(f" - Version: {version.version}, Stage: {version.current_stage}, Status: {version.status}")
