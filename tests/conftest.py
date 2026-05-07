import mlflow
import os


def pytest_configure(config):
    """Khởi tạo MLflow tracking trước khi chạy bất kỳ test nào."""
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow_test.db")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("test-experiment")
