from monitorability_eval.config import ASSETS, dataset_manifest
TASK_PATH_MAPPING = {task: str(ASSETS / spec["path"]) for task, spec in dataset_manifest().items()}
