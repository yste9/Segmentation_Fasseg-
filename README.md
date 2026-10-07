# Segmentation_Fasseg-
Segmentation sémantique de visages sur le dataset FASSEG avec U-Net et PyTorch : classification pixel par pixel en 9 classes, comparaison de deux modèles et évaluation quantitative et visuelle.


# Segmentation sémantique de visages avec U-Net

Projet de Computer Vision réalisé sur le dataset **FASSEG**, avec **Python et PyTorch**.

L’objectif est d’attribuer une classe à chaque pixel d’une image afin d’identifier les différentes régions d’un visage.

**Auteur : Yannick ASSI**  
**Cadre : projet universitaire — Master MIASHS, UCO Angers**  
**Date : mars 2026**

---

## 1. Étude à réaliser

### 🎯 Objectif du projet

Développer un modèle de segmentation sémantique fondé sur un réseau de neurones convolutionnel, en utilisant une architecture **U-Net**.

Contrairement à la classification, qui attribue une étiquette à une image entière, la segmentation sémantique produit une prédiction pour chaque pixel.

Dans ce projet, chaque pixel doit être affecté à l’une des **9 classes** du dataset FASSEG.

### 🗂️ Organisation des données

Les données sont réparties dans quatre dossiers :

| Dossier | Contenu | Utilisation |
|---|---|---|
| `Train_RGB` | Images d’apprentissage | Constitution des ensembles d’entraînement et de validation |
| `Train_Labels` | Masques associés aux images d’apprentissage | Annotations pour l’entraînement et la validation |
| `Test_RGB` | Images de test | Évaluation finale |
| `Test_Labels` | Masques associés aux images de test | Comparaison avec les prédictions finales |

Les images et leurs masques doivent être correctement associés.

Les données d’apprentissage doivent être séparées en deux ensembles :

- **Entraînement** : apprentissage des paramètres du modèle.
- **Validation** : suivi de l’apprentissage et choix de la configuration.

Une répartition **80 % / 20 %** est proposée dans les consignes.

**Le jeu de test doit être réservé exclusivement à l’évaluation finale.**

### 🏷️ Classes à prédire

| Identifiant | Classe originale | Description |
|---|---|---|
| 0 | Background | Arrière-plan |
| 1 | Hair | Cheveux |
| 2 | Face | Visage |
| 3 | Right eyebrow | Sourcil droit |
| 4 | Left eyebrow | Sourcil gauche |
| 5 | Right eye | Œil droit |
| 6 | Left eye | Œil gauche |
| 7 | Nose | Nez |
| 8 | Mouth | Bouche |

### ⚙️ Travail à effectuer

#### Étape 1 — Explorer les données

- Examiner les images et leurs masques.
- Vérifier la correspondance entre chaque image et son annotation.
- Identifier les valeurs représentant les 9 classes.
- Visualiser plusieurs couples image / masque pour comprendre les annotations.

#### Étape 2 — Préparer les données

- Constituer les ensembles d’entraînement et de validation à partir des données d’apprentissage.
- Harmoniser les dimensions des images et des masques.
- Transformer les images en tenseurs adaptés au modèle.
- Préparer les masques comme des cartes d’étiquettes entières.
- Conserver les identifiants des classes lors des transformations.

#### Étape 3 — Construire le modèle U-Net

- Définir un réseau convolutionnel adapté à la segmentation.
- Mettre en place un encodeur pour extraire les caractéristiques.
- Mettre en place un décodeur pour reconstruire une prédiction spatiale.
- Utiliser des connexions entre encodeur et décodeur pour conserver les détails.
- Produire des scores pour les **9 classes à chaque pixel**.

#### Étape 4 — Entraîner et valider le modèle

- Choisir une fonction de perte adaptée à la classification multiclasse.
- Entraîner le réseau sur l’ensemble d’entraînement.
- Suivre les performances sur l’ensemble de validation.
- Observer l’évolution des pertes.
- Ajuster la configuration à partir des résultats de validation.

#### Étape 5 — Évaluer quantitativement

Évaluer le modèle final sur les données de test.

La métrique demandée dans les consignes est la **Pixel Accuracy** :

> Proportion de pixels dont la classe prédite correspond à la classe réelle.

#### Étape 6 — Évaluer qualitativement

Présenter des visualisations comparant :

1. L’image originale.
2. Le masque réel, ou vérité terrain.
3. Le masque prédit par le modèle.

Analyser les régions correctement segmentées, les confusions entre classes et les difficultés sur les petites zones du visage.

### 📋 Résultats attendus

- Un modèle U-Net entraîné.
- Une évaluation finale sur le jeu de test.
- La Pixel Accuracy obtenue.
- Des comparaisons visuelles entre images, annotations et prédictions.
- Une analyse des performances et des limites.

---

## 2. Réalisation et résultats obtenus

Les éléments de cette partie sont rapportés dans le document de présentation du projet.

### 📊 Données utilisées

Le rapport indique :

- **20 images d’apprentissage**, à utiliser pour l’entraînement et la validation.
- **50 images de test**.
- **9 classes de segmentation**.

La petite taille de l’ensemble d’apprentissage constitue une contrainte importante du projet.

### 🔧 Prétraitement réalisé

Les transformations suivantes ont été appliquées :

- Redimensionnement en **256 × 256 pixels**.
- Normalisation des valeurs des images.
- Conversion en tenseurs PyTorch.
- Redimensionnement des masques par interpolation **au plus proche voisin**.

L’interpolation au plus proche voisin permet de conserver les étiquettes discrètes des masques, sans créer de valeurs intermédiaires entre les classes.

### 🧪 Première expérimentation

Une première version a permis de vérifier le fonctionnement de la chaîne de traitement.

| Élément | Configuration |
|---|---|
| Architecture | U-Net simple |
| Fonction de perte | `CrossEntropyLoss` |
| Durée d’entraînement | 10 epochs |
| Pixel Accuracy | 70,35 % |

Les prédictions restaient approximatives :

- Confusions entre certaines classes.
- Segmentation imprécise des régions fines.
- Masques prédits encore éloignés de la vérité terrain.

Cette expérimentation a servi de référence pour la seconde version.

### 🚀 Amélioration du modèle

La seconde version comporte plusieurs modifications :

- Une architecture **U-Net plus profonde**.
- L’ajout de **Batch Normalization**.
- Une **fonction de perte pondérée**.
- Un entraînement prolongé à **40 epochs**.

Ces changements visent à améliorer l’apprentissage et la représentation des différentes régions du visage.

### 📉 Analyse de l’apprentissage

Le rapport présente les courbes de perte d’entraînement et de validation.

Une diminution progressive des pertes est observée, ce qui indique que le modèle améliore sa représentation des données au cours de l’apprentissage.

Cette observation doit être complétée par l’évaluation sur le jeu de test pour apprécier sa capacité de généralisation.

### 📈 Performances finales

Le modèle amélioré a été évalué sur le jeu de test.

| Métrique | Valeur |
|---|---:|
| Pixel Accuracy | **90,88 %** |
| Mean Intersection over Union — mIoU | **0,4912** |

#### Lecture des métriques

**Pixel Accuracy : 90,88 %**

Le modèle attribue la bonne classe à environ neuf pixels sur dix sur le jeu de test.

**Mean IoU : 0,4912**

Cette métrique mesure le recouvrement entre les régions prédites et les régions réelles, puis en fait la moyenne sur les classes prises en compte.

Elle complète la Pixel Accuracy : un score global élevé peut masquer des difficultés sur les petites régions.

### 🔄 Comparaison des deux versions

| Modèle | Epochs | Pixel Accuracy | Analyse visuelle |
|---|---:|---:|---|
| U-Net initial | 10 | 70,35 % | Segmentation approximative |
| U-Net amélioré | 40 | 90,88 % | Segmentation plus précise et cohérente |

La Pixel Accuracy progresse de **20,53 points de pourcentage**.

Plusieurs paramètres ayant été modifiés simultanément, cette comparaison ne permet pas d’isoler la contribution de chaque modification.

### 👁️ Analyse qualitative

Les exemples présentés dans le rapport montrent une meilleure restitution des principales régions :

- Contour du visage.
- Cheveux.
- Yeux.
- Nez.
- Bouche.

Les prédictions du modèle amélioré sont plus cohérentes spatialement et plus proches des masques réels.

Des erreurs persistent principalement sur :

- Les sourcils.
- Les petites régions.
- Certains contours et frontières entre classes.

Ces observations portent sur les exemples visualisés dans le rapport. Une évaluation par classe permettrait de préciser leur fréquence sur l’ensemble du test.

---

## 3. Enseignements et limites

### Enseignements

- Le prétraitement des masques est essentiel pour préserver les classes.
- La configuration de l’architecture influence la qualité des prédictions.
- L’analyse visuelle complète utilement les métriques.
- La Pixel Accuracy doit être interprétée avec des mesures tenant compte des différentes classes.

### Limites

- Ensemble d’apprentissage très réduit.
- Difficultés persistantes sur les régions fines.
- Absence de résultats détaillés par classe dans le rapport.
- Répartition exacte entraînement / validation et paramètres complets à documenter dans le code.

---

## 4. Perspectives

Les pistes proposées dans le rapport sont :

- **Augmentation de données**.
- **Utilisation de modèles pré-entraînés**.
- **Amélioration des performances par classe**.

Pour approfondir l’évaluation, il serait également utile de :

- Présenter l’IoU de chaque classe.
- Documenter les paramètres et la graine aléatoire.
- Comparer les modifications une par une.
- Ajouter les instructions permettant de reproduire l’entraînement et l’évaluation.

---

## 5. Technologies et compétences

| Domaine | Technologies et concepts |
|---|---|
| Programmation | Python |
| Deep Learning | PyTorch, réseaux convolutionnels |
| Architecture | U-Net |
| Prétraitement | Redimensionnement, normalisation, tenseurs |
| Apprentissage | CrossEntropyLoss, pondération, Batch Normalization |
| Évaluation | Pixel Accuracy, Mean IoU, analyse qualitative |
| Application | Segmentation sémantique de visages |

---

## 6. Auteur

**Yannick ASSI**

Projet universitaire — Master MIASHS  
Université Catholique de l’Ouest
