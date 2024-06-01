import json


def save_metadata(hyperparams: dict, metrics: dict, metadata_path: str):
    metadata = {
        'hyperparameters': hyperparams,
        'metrics': metrics
    }
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f)
    print(f"Métadonnées enregistrées dans {metadata_path}")