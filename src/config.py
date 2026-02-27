from pathlib import Path

SCREENERS_LIST = ['toxicity', 'solubility_will', 'solubility_jana']
DEVICE_OPTIONS = ['cpu','cuda','mps']

FOLDER_SIGNATURE = 'screening_run_XX'
OUTPUT_DIR = Path('static/runs')

toxicity_clf_path = Path('src/screeners/toxicity/RFC_esm.pkl')
solubility_will_clf_path = Path('src/screeners/solubility/williams_model.joblib')
solubility_jana_clf_path = Path('src/screeners/solubility/janas_model.pkl')