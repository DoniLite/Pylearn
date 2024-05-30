# Chemin vers l'environnement virtuel
VENV := env

# Commande pour créer l'environnement virtuel
create_venv:
	python -m venv $(VENV)

# Commande pour activer l'environnement virtuel et installer les dépendances
install_deps: create_venv
	$(VENV)/bin/pip install -r requirements.txt

# Commande pour lancer l'application
run:
	$(VENV)/bin/python app.py

# Commande pour nettoyer les fichiers générés
clean:
	rm -rf $(VENV) __pycache__ *.pyc *.pyo
