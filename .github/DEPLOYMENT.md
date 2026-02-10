# 🚀 Déploiement Automatique Docker - ShroomLoc

Ce repository est configuré avec GitHub Actions pour construire et déployer automatiquement l'image Docker de ShroomLoc.

## 📋 Fonctionnalités du Workflow

- ✅ **Tests automatiques** sur chaque push/PR
- 🏗️ **Build multi-architecture** (AMD64 + ARM64)
- 📦 **Publication sur GitHub Container Registry** (GHCR)
- 🏷️ **Tagging intelligent** basé sur les branches et tags Git
- 🚀 **Déploiement automatique** sur la branche master

## 🔧 Configuration (Aucun Secret Requis)

Le workflow utilise GitHub Container Registry (GHCR) qui s'authentifie automatiquement avec `GITHUB_TOKEN` - aucune configuration supplémentaire n'est nécessaire !

### Images Disponibles

Après chaque déploiement, votre image sera disponible à :
```
ghcr.io/shrooml-team/shroomloc:latest
ghcr.io/shrooml-team/shroomloc:<branch-name>
ghcr.io/shrooml-team/shroomloc:v<version> # pour les tags
```

## 🚀 Utilisation des Images

### Pull et Run Local
```bash
# Dernière version stable
docker pull ghcr.io/shrooml-team/shroomloc:latest
docker run -p 8000:8000 ghcr.io/shrooml-team/shroomloc:latest

# Version spécifique
docker pull ghcr.io/shrooml-team/shroomloc:v1.0.0
docker run -p 8000:8000 ghcr.io/shrooml-team/shroomloc:v1.0.0
```

### Docker Compose
```yaml
version: '3.8'
services:
  shroomloc:
    image: ghcr.io/shrooml-team/shroomloc:latest
    ports:
      - "8000:8000"
    restart: unless-stopped
```

## 🔄 Déclenchement du Workflow

Le workflow se déclenche automatiquement :
- ✅ Sur chaque **push** vers `master`
- ✅ Sur chaque **tag** commençant par `v` (ex: `v1.0.0`)
- ✅ Sur chaque **Pull Request** (build seulement, pas de déploiement)

### Créer une Release
```bash
# Créer et pousser un tag
git tag v1.0.0
git push origin v1.0.0

# Cela déclenchera automatiquement le build et le déploiement
```

## 📊 Status du Workflow

![Build Status](https://github.com/ShrooML-Team/ShroomLoc/workflows/Build%20and%20Deploy%20Docker/badge.svg)

Vous pouvez suivre le status des builds dans l'onglet **Actions** de votre repository GitHub.

## 🔧 Configuration Avancée (Optionnel)

### Déploiement Automatique sur Serveur

Pour déployer automatiquement sur un serveur, décommentez et configurez la section `Deploy to server` dans le workflow, puis ajoutez ces secrets dans votre repository :

- `HOST` : Adresse IP ou nom de domaine de votre serveur
- `USERNAME` : Nom d'utilisateur SSH
- `SSH_KEY` : Clé privée SSH pour l'authentification

### Variables d'Environnement

Si votre application nécessite des variables d'environnement, vous pouvez les ajouter dans le workflow ou utiliser GitHub Secrets.

## 🐛 Troubleshooting

### "Package not found" lors du pull
Assurez-vous que votre repository GitHub a les permissions correctes pour GHCR :
1. Allez dans **Settings** → **Actions** → **General**
2. Sous "Workflow permissions", sélectionnez **Read and write permissions**

### Build qui échoue
Vérifiez les logs dans l'onglet **Actions** pour identifier le problème :
- Erreurs de syntax dans le Dockerfile
- Tests qui échouent
- Problèmes de dépendances

## 📝 Logs et Monitoring

Les logs de déploiement sont disponibles dans l'onglet **Actions** de GitHub. Pour des logs d'application en temps réel, connectez-vous à votre serveur ou utilisez les outils de monitoring de votre infrastructure.