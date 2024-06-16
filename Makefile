# Chemin vers l'environnement virtuel
VENV := envs
PYTHON := $(VENV)/bin/python

# Commande pour créer l'environnement virtuel
create_venv:
	python -m venv $(VENV)

# Commande pour activer l'environnement virtuel et installer les dépendances
install_deps: create_venv
	$(VENV)/bin/pip install -r requirements.txt

# Commande pour lancer l'application
run:
	$(VENV)/bin/python app.py

# Commande pour entraîner le modèle
train:
	$(PYTHON) learning/train_model.py

#
preprocess:
	$(PYTHON) learning/preprocess.py

# Commande pour filtrer les données
filter:
	$(PYTHON) learning/filter_data.py

# Commande pour nettoyer les fichiers générés
clean:
	rm -rf $(VENV) __pycache__ *.pyc *.pyo

freeze:
	pip freeze > requirements.txt
