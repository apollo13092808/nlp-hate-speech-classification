import logging
import os
from pathlib import Path


logging.basicConfig(format="[%(asctime)s]: %(message)s:", level=logging.INFO)


project_name = "classifier"

list_of_files = [
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/data_ingestion.py",
    f"src/{project_name}/components/data_transformation.py",
    f"src/{project_name}/components/model_evaluation.py",
    f"src/{project_name}/components/model_training.py",
    f"src/{project_name}/components/model_uploading.py",
    f"src/{project_name}/configurations/__init__.py",
    f"src/{project_name}/configurations/syncer.py",
    f"src/{project_name}/constants/__init__.py",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/entity/artifacts.py",
    f"src/{project_name}/entity/configs.py",
    f"src/{project_name}/exception/__init__.py",
    f"src/{project_name}/logger/__init__.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/pipeline/training.py",
    f"src/{project_name}/pipeline/prediction.py",
    f"src/{project_name}/ml/__init__.py",
    f"src/{project_name}/ml/model.py",
    "app.py",
    "demo.py",
    "setup.py",
    "requirements.txt",
    "Dockerfile",
    ".dockerignore",
]

for filepath in list_of_files:
    filepath = Path(filepath)
    dir, name = os.path.split(p=filepath)

    if dir != "":
        os.makedirs(name=dir, exist_ok=True)
        logging.info(msg=f"Creating directory... {dir} for the file: {name}")

    if (not os.path.exists(path=filepath)) or (os.path.getsize(filename=filepath) == 0):
        with open(filepath, "w") as f:
            pass
            logging.info(msg=f"Creating empty file: {filepath}")
    else:
        logging.info(msg=f"{name} is already exists")
