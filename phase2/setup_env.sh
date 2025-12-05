#!/bin/bash

# Script qui lance un nouveau shell avec le venv activé
# Utilisation: ./activate_venv.sh

VENV_DIR="venv"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="$SCRIPT_DIR/$VENV_DIR"

REQUIREMENT_FILE="$SCRIPT_DIR/requirement.txt"

# Vérifier si le venv existe déjà
if [ ! -d "$VENV_PATH" ]; then
    echo "Création de l'environnement virtuel..."
    python3 -m venv "$VENV_PATH"
    
    if [ $? -ne 0 ]; then
        echo "✗ Erreur lors de la création de l'environnement virtuel"
        exit 1
    fi
    echo "✓ Environnement virtuel créé avec succès"
fi

# Mettre à jour pip3 vers la dernière version
echo "Mise à jour de pip3 vers la dernière version..."
"$VENV_PATH/bin/pip3" install --upgrade pip --quiet
if [ $? -eq 0 ]; then
    echo "✓ pip3 mis à jour avec succès"
else
    echo "⚠ Avertissement: Impossible de mettre à jour pip3 (continuation avec la version actuelle)"
fi

# Installer ou mettre à jour les dépendances dans le venv
if [ -f "$REQUIREMENT_FILE" ]; then
    echo "Installation des dépendances dans le venv..."
    "$VENV_PATH/bin/pip3" install -r "$REQUIREMENT_FILE"
    if [ $? -eq 0 ]; then
        echo "✓ Dépendances installées avec succès"
    else
        echo "✗ Erreur lors de l'installation des dépendances"
        exit 1
    fi
fi

# Lancer un nouveau shell avec le venv activé
echo "Lancement d'un nouveau shell avec le venv activé..."
echo "Tapez 'exit' pour quitter le shell et revenir à votre shell précédent."
echo ""

exec $SHELL -c "source '$VENV_PATH/bin/activate' && cd '$SCRIPT_DIR' && exec $SHELL"

