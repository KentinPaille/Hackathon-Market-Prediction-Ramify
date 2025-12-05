# Guide de Soumission - Phase 1

Ce document explique comment soumettre votre bot de trading sur la plateforme du hackathon.

## 📦 Préparation du Fichier de Soumission

### Créer un Archive ZIP

Vous devez créer un fichier ZIP contenant votre fichier `bot_trade.py`. Voici les commandes pour créer l'archive :

#### Sur Linux/macOS :

```bash
# Depuis le dossier phase_1
zip submission.zip bot_trade.py
```

### Inclure des Fichiers Supplémentaires

Si votre fichier `bot_trade.py` dépend d'autres fichiers Python (modules personnalisés, utilitaires, etc.), vous pouvez les inclure dans le même ZIP :

#### Exemple avec plusieurs fichiers :

```bash
# Sur Linux/macOS - Inclure bot_trade.py et d'autres fichiers
zip submission.zip bot_trade.py utils.py models.py

# Ou inclure tous les fichiers Python d'un dossier
zip submission.zip bot_trade.py helpers/*.py
```

**Important :**
- ✅ Le fichier `bot_trade.py` doit être à la racine du ZIP (pas dans un sous-dossier)
- ✅ Tous les fichiers Python supplémentaires doivent être accessibles depuis `bot_trade.py`
- ✅ N'incluez **PAS** les fichiers de données (CSV), le venv, ou les fichiers de configuration locaux
- ✅ N'incluez **PAS** le fichier `main.py` ou les fichiers du dossier `scoring/` (déjà présents sur la plateforme)

## 🌐 Soumission sur la Plateforme

### 1. Accéder à la Plateforme

Rendez-vous sur la plateforme du hackathon :

**URL :** https://hackathon-x-poc.ramify.fr

### 2. Se Connecter

- Connectez-vous avec le SSO Discord

### 3. Remplir le Formulaire de Soumission

Une fois connecté :

1. Accédez à la section de soumission
2. Remplissez le formulaire de soumission avec les informations suivantes :
   - **Nom du bot** : Donnez un nom à votre bot
   - **Fichier ZIP** : Uploadez votre fichier `submission.zip`
3. Soumettez le formulaire

### 4. Confirmation

Après la soumission, vous devriez recevoir une confirmation que votre bot a été reçu et est en attente d'exécution.

## 📊 Consulter les Résultats

Le dashboard affiche :

- **📈 Scores de Performance** :
  - Sharpe Score
  - PnL Score
  - Max Drawdown Score
  - Base Score (score global)

- **📋 Logs d'Exécution** :
  - Logs détaillés de l'exécution de votre bot
  - Erreurs éventuelles (si la soumission a échoué)
  - Messages de validation

- **⏱️ Statut** :
  - Statut de la soumission (en attente, en cours, terminé, erreur)
  - Date et heure de soumission
  - Date et heure d'exécution

## ⚠️ Points Importants

### Avant de Soumettre

- ✅ Testez votre bot localement avec `python3 main.py data/asset_a_test.csv`
- ✅ Vérifiez que votre fonction `make_decision` respecte la signature exacte
- ✅ Assurez-vous que le format de retour est correct (dictionnaire avec 'Asset A' et 'Cash')
- ✅ Si vous utilisez des fichiers supplémentaires, testez qu'ils fonctionnent ensemble

## 📞 Support

En cas de problème lors de la soumission ou pour toute question, contactez l'équipe du hackathon via discord

