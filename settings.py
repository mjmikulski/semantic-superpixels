import os

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
DATA_ROOT = os.path.join(PROJECT_ROOT, 'data')
IMAGES_DIR = os.path.join(DATA_ROOT, 'img')
MODELS_DIR = os.path.join(DATA_ROOT, 'models')
ARTEFACTS_DIR = os.path.join(DATA_ROOT, 'artefacts')

# Create directories based on constants that end with '_DIR' defined in this file.
DIRECTORIES_TO_CREATE = [v for k, v in globals().copy().items() if k.endswith('_DIR')]
for directory in DIRECTORIES_TO_CREATE:
    os.makedirs(directory, exist_ok=True)
