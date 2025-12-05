# Guide de Soumission - Phase 1

Ce document explique comment créer et soumettre votre bot de trading pour la Phase 1 du hackathon.

## 📋 Structure Requise

### Fichier `bot_trade.py`

Vous devez créer un fichier nommé **`bot_trade.py`** à la racine du dossier `phase_1/`. Ce fichier doit contenir une fonction obligatoire avec la signature exacte suivante :

```python
def make_decision(epoch: int, price: float):
    """
    Fonction principale qui détermine l'allocation du portefeuille à chaque époque.
    
    Parameters
    ----------
    epoch : int
        L'époque (index temporel) actuelle dans la série de données
    price : float
        Le prix actuel de l'asset 'Asset A'
    
    Returns
    -------
    dict
        Un dictionnaire contenant la répartition du portefeuille entre les assets.
        Les clés doivent être exactement 'Asset A' et 'Cash'.
        Les valeurs doivent être des nombres entre 0 et 1, et leur somme doit être égale à 1.0.
        
    Example
    -------
    >>> make_decision(0, 100.5)
    {'Asset A': 0.3, 'Cash': 0.7}
    """
    # Votre logique de trading ici
    return {'Asset A': 0.3, 'Cash': 0.7}
```

### Format de Retour

La fonction `make_decision` doit retourner un dictionnaire Python avec les caractéristiques suivantes :

- **Clés obligatoires** : `'Asset A'` et `'Cash'` (exactement ces noms)
- **Valeurs** : Des nombres flottants ou entiers entre 0 et 1 (inclus)
- **Somme** : La somme des valeurs doit être exactement égale à 1.0

**Exemples valides :**
```python
{'Asset A': 0.3, 'Cash': 0.7}      # 30% dans Asset A, 70% en Cash
{'Asset A': 1.0, 'Cash': 0.0}      # 100% dans Asset A, 0% en Cash
{'Asset A': 0.0, 'Cash': 1.0}      # 0% dans Asset A, 100% en Cash
{'Asset A': 0.5, 'Cash': 0.5}      # 50% dans Asset A, 50% en Cash
```

**Exemples invalides :**
```python
{'Asset A': 0.3, 'Cash': 0.6}      # ❌ Somme = 0.9 (doit être 1.0)
{'Asset A': 0.3, 'Cash': 0.8}      # ❌ Somme = 1.1 (doit être 1.0)
{'Asset': 0.5, 'Cash': 0.5}        # ❌ Clé incorrecte (doit être 'Asset A')
{'Asset A': -0.1, 'Cash': 1.1}     # ❌ Valeurs hors limites [0, 1]
```

## 🧪 Tester Votre Bot

### Commande de Test

Pour tester votre bot, utilisez le programme de test fourni par Ramify :

```bash
python3 main.py data/asset_a_test.csv
```

**Arguments :**
- **Premier argument** : Le fichier `main.py` (exécuté directement)
- **Deuxième argument** : Le chemin vers le dataset de test (ex: `data/asset_a_test.csv`)

### Afficher le Graphique de Performance

Pour visualiser un graphique représentant la performance de votre bot, ajoutez le paramètre `--show-graph` :

```bash
python3 main.py data/asset_a_test.csv --show-graph
```

Le graphique affichera :
- L'évolution du PnL (Profit and Loss) au fil du temps
- Les zones de profit (vert) et de perte (rouge)
- La ligne de référence du capital initial

### Résultats Affichés

Lors de l'exécution, le programme affichera :

1. **Scores** :
   - Sharpe Score
   - PnL Score
   - Max Drawdown Score
   - Base Score (score global)

2. **Graphique** (si `--show-graph` est utilisé) :
   - Courbe d'évolution du PnL
   - Visualisation des performances

## 📦 Setup de l'Environnement

Pour configurer l'environnement de développement avec toutes les dépendances nécessaires, il suffit d'exécuter le script shell fourni :

**Important** : Avant de pouvoir exécuter le script, vous devez le rendre exécutable avec la commande `chmod` :

```bash
chmod +x setup_env.sh
```

Ensuite, exécutez le script :

```bash
./setup_env.sh
```

Ce script va :
1. Créer un environnement virtuel Python (s'il n'existe pas déjà)
2. Installer automatiquement toutes les bibliothèques requises depuis `requirement.txt`
3. **Démarrer un nouveau shell** avec l'environnement virtuel activé

Une fois le nouveau shell lancé, vous aurez accès à toutes les bibliothèques installées :
- `matplotlib` : Pour l'affichage des graphiques
- `pandas` : Pour la manipulation des données
- `numpy` : Pour les calculs numériques

**Note** : Pour quitter le shell avec l'environnement activé, tapez simplement `exit` pour revenir à votre shell précédent.

## ⚠️ Validation

Le programme de test valide automatiquement votre fonction `make_decision` :

- ✅ Vérification des clés du dictionnaire
- ✅ Vérification que les valeurs sont numériques
- ✅ Vérification que les valeurs sont entre 0 et 1
- ✅ Vérification que la somme des allocations est égale à 1.0

Si une validation échoue, une erreur explicite sera affichée avec les détails du problème.

## 💡 Exemple de Bot Simple

Voici un exemple minimal de `bot_trade.py` :

```python
def make_decision(epoch: int, price: float):
    """
    Exemple simple : allocation fixe 50/50
    """
    return {'Asset A': 0.5, 'Cash': 0.5}
```

## 📝 Notes Importantes

1. **Nom du fichier** : Le fichier doit s'appeler exactement `bot_trade.py`

2. **Nom de la fonction** : La fonction doit s'appeler exactement `make_decision` (respecter la casse)

3. **Signature** : La signature doit être exactement `def make_decision(epoch: int, price: float):`

4. **Format de retour** : Le dictionnaire doit contenir exactement les clés `'Asset A'` et `'Cash'`

5. **Somme des allocations** : La somme des valeurs doit être exactement 1.0 (tolérance de 0.00001)

6. **Historique** : Vous pouvez maintenir un historique des prix dans votre fichier pour implémenter des stratégies basées sur l'historique

## 🚀 Prochaines Étapes

Une fois votre bot testé localement et validé, vous pouvez le soumettre via la plateforme du hackathon. Le même système de validation sera utilisé lors de la soumission officielle.

