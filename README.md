# Auto MPG · Vehicle Lab

Application Streamlit autonome : les 398 véhicules fournis et les 6 puissances
manquantes sont intégrés dans `app.py` via `StringIO`. Aucun CSV externe ni
connexion Internet n'est nécessaire après l'installation des dépendances.

## Démarrer

Utiliser de préférence Python 3.11 ou 3.12 dans un environnement dédié.
Ouvrir un terminal dans le dossier extrait `auto_mpg_app`.
Si votre environnement est déjà activé, passer directement à l'installation.

Sous Windows PowerShell, créer et activer un environnement si nécessaire :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Installer la base et lancer l'application :

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

La commande `streamlit run app.py` fonctionne également lorsque l'environnement
est activé. Ouvrir l'adresse locale indiquée par Streamlit.

## Activer la RTX 3050 sous Windows : XGBoost CUDA

cuML n'est pas distribué pour Windows natif : il demande Linux ou WSL2.
L'application utilise donc automatiquement XGBoost CUDA si cuML est absent et si
CUDA + CuPy sont fonctionnels. Cette alternative est un boosting d'arbres ; elle
n'est pas un Random Forest et son nom est affiché dans l'interface.

Pour le profil GPU proposé ci-dessous, installer un pilote NVIDIA récent
compatible avec CUDA 12 et le CUDA Toolkit 12.x. Ces composants système ne sont
pas installés par `requirements.txt`. Installer également Visual C++
Redistributable x64 si XGBoost signale une DLL manquante.

Dans le même environnement que celui qui lance Streamlit :

```powershell
nvidia-smi
python -m pip install -r requirements-gpu-cuda12.txt
python -c "import cupy as cp; print(cp.cuda.runtime.getDeviceProperties(0)['name']); print((cp.ones(4)*2).sum().item())"
python -m streamlit run app.py
```

Le profil GPU fixe XGBoost 2.1.4 et CuPy 13.6.0 pour cibler CUDA 12 avec Python
3.11/3.12. Ce ne sont pas des contraintes imposées par `app.py`, qui accepte
également des versions plus récentes compatibles avec votre pile CUDA.
N'installer qu'une seule distribution de CuPy dans un environnement.

Si vous utilisez déjà Conda et ne souhaitez pas installer de Toolkit système,
une autre voie consiste à créer un environnement dédié contenant le runtime :

```powershell
conda create -n auto-mpg-gpu -c conda-forge python=3.11 cupy=13.6 cuda-version=12.8
conda activate auto-mpg-gpu
python -m pip install -r requirements.txt xgboost==2.1.4
python -m streamlit run app.py
```

Le pilote NVIDIA du système reste indispensable. Pour une erreur de pilote,
de runtime ou de version, consulter « Diagnostic des moteurs » dans la barre
latérale : le motif exact du repli CPU y est conservé.

## Garder un Random Forest sur GPU : cuML sous WSL2/Linux

Pour un véritable RandomForestRegressor CUDA, utiliser un environnement
WSL2/Linux avec le pilote NVIDIA compatible, puis installer cuML et CuPy à
l'aide du sélecteur officiel RAPIDS :

https://docs.rapids.ai/install/

Choisir les versions Python, CUDA et RAPIDS compatibles avec le système.
Installer les dépendances de base dans cet environnement puis lancer :

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Lorsqu'un cuML fonctionnel est présent, il est prioritaire sur XGBoost.
La présence d'une RTX 3050 dans le PC ne suffit pas à rendre CUDA disponible dans
n'importe quel environnement Python : le pilote et les bibliothèques doivent
être compatibles et installés dans l'environnement qui exécute l'application.

## Ordre de sélection du moteur

1. Random Forest cuML : entraînement et prédiction sur tableaux CuPy.
2. XGBoost `tree_method='hist', device='cuda:0'` : entraînement sur CuPy et
   prédiction par `inplace_predict` sur CuPy. La configuration du booster est
   contrôlée pour détecter les replis silencieux sur CPU.
3. Random Forest scikit-learn : CPU si CUDA ou les deux moteurs GPU échouent.

Une allocation et une opération CUDA sont testées avant les moteurs GPU.
Le nom du GPU réel est affiché, au lieu de supposer que tout serveur possède
une RTX 3050. Le statut actif n'apparaît qu'après entraînement et prédiction de
test réussis. Une erreur GPU ultérieure déclenche un réentraînement CPU sur le
même split et recalcule automatiquement la prédiction demandée.
Le bouton « Réessayer les moteurs GPU » relance la détection.

Sur un serveur distant, c'est le GPU **du serveur** qui est utilisé, pas celui
du navigateur. Une machine d'hébergement sans CUDA passe sur CPU.

## Les quatre onglets

- **Analyse descriptive** : filtres origine/année/cylindres/recherche littérale,
  cinq KPIs, nuage poids-consommation, évolution annuelle, répartition et
  boxplots par origine, top/flop 5, tableau et téléchargement CSV.
- **Comparateur** : sélection de deux observations distinctes dans le catalogue
  complet, caractéristiques côte à côte, écarts de consommation, puissance et CO₂.
  Les noms répétés sont distingués par année et identifiant.
- **Prédiction & carbone** : modèle, R², RMSE, chronométrage, importance par
  permutation, configuration personnalisée et CO₂ annuel.
- **Données & diagnostics** : audit, unités, méthode, comparaison à une référence
  naïve, observations/prédictions et résidus du test, export du CSV brut.

La demande décrit trois onglets malgré un total annoncé de quatre ; le dernier
complète l'application par des diagnostics utiles.

## Méthode et limites

- Par défaut, les filtres s'appliquent aux analyses **et au jeu d'apprentissage**.
  Décocher « Entraîner sur la sélection filtrée » pour conserver les 398 véhicules
  pour le ML. Le comparateur garde toujours le catalogue complet.
- L'application accepte les filtres vides. Elle exige au moins 30 observations
  et deux valeurs de cible différentes pour entraîner ; les autres fonctions
  restent accessibles lorsqu'une sélection est trop petite.
- Pour les statistiques descriptives, la médiane de puissance est 93,5 hp.
- Pour le ML, le split a lieu **avant** l'imputation. Les médianes ne sont
  apprises que sur le train ; l'origine est encodée en one-hot.
- Les entrées sont cylindres, cylindrée, puissance, poids, accélération, année et
  origine. Le MPG cible et ses dérivés, le nom et la marque ne sont pas des entrées.
- Graine 42, split aléatoire, 80 % / 20 % par défaut. Les versions proches d'un
  même modèle peuvent être présentes dans les deux ensembles : ce n'est pas une
  validation par famille ou par année, ni une garantie sur des véhicules modernes.
- Les paramètres peuvent être ajustés dans la barre latérale. Le modèle est
  conservé par session et réentraîné uniquement si sa sélection ou ses paramètres
  changent, si le GPU échoue, ou sur demande explicite de nouvel essai GPU.
- La mesure d'entraînement couvre `fit` et la synchronisation CUDA ; elle exclut
  le prétraitement et le transfert initial train vers GPU. La prédiction de test
  inclut les transferts. Le premier appel peut inclure une initialisation du moteur.
  Il ne s'agit pas d'un benchmark répété avec échauffement et mesure de variance.
- Les importances correspondent à la hausse de RMSE après permutation sur le
  test (3 répétitions), avec permutation groupée des colonnes one-hot d'origine.
  Les variables corrélées peuvent se partager l'importance. Une valeur négative
  est possible. Ces diagnostics ne doivent pas servir à ajuster indéfiniment le
  modèle sur ce même test.
- Un GPU n'est pas nécessairement plus rapide pour seulement 398 observations.
- `weight` est en **lb**, `displacement` en **in³**, `horsepower` en **hp**,
  `acceleration` est le temps 0–60 mph en secondes et le MPG est le **MPG US**.
- `model_year` contient les codes 70–82. L'interface affiche 1970–1982.
- La marque est le premier mot de `name`, sans correction des variantes du fichier.
  Les « origines » sont des régions ; « europe » n'est pas un pays.
- La consommation dérivée est `235.215 / mpg`. La moyenne des consommations en
  L/100 km est calculée individuellement puis moyennée ; elle ne correspond pas
  à la conversion du MPG moyen.
- Le CO₂ demandé est `L/100 km × 23.5` g/km. L'estimation annuelle en tonnes est
  `g/km × km/an / 1_000_000`. Le coefficient uniforme correspond à une hypothèse
  proche de l'essence ; le dataset inclut aussi des diesels. Ce calcul pédagogique
  n'est pas une mesure certifiée, ne distingue pas les carburants et ne couvre
  pas le cycle de vie.

## Vérification locale

```powershell
python verify_app.py
```

Le script vérifie les données, le split et l'imputation, les chemins de repli
simulés, les conversions, ainsi que les interactions Streamlit. Les simulations
ne remplacent pas un test matériel sur la RTX 3050.

Lors de la préparation, le chemin CPU a été exécuté avec les réglages par défaut :
318 véhicules de train / 80 de test, **R² ≈ 0,917**, **RMSE ≈ 2,11 MPG**.
Les 8 vérifications de `verify_app.py` ont réussi. Environnement de vérification :
Python 3.12, Streamlit 1.66.1, pandas 2.2.3, NumPy 2.3.5, Plotly 7.1.0 et
scikit-learn 1.8.0.

Aucun GPU CUDA n'était disponible dans l'environnement de préparation ; les
chemins matériels cuML/XGBoost doivent être validés sur le PC NVIDIA.

## Documentation officielle

- RAPIDS, compatibilité Windows/WSL2 : https://docs.rapids.ai/install/
- cuML Random Forest : https://docs.nvidia.com/cuml/26.08/api/generated/cuml.ensemble.RandomForestRegressor/
- XGBoost GPU : https://xgboost.readthedocs.io/en/stable/gpu/
- XGBoost installation : https://xgboost.readthedocs.io/en/stable/install.html
- CuPy installation : https://docs.cupy.dev/en/stable/install.html

Les versions et exigences CUDA évoluent : le sélecteur RAPIDS et la documentation
de la version installée priment pour la compatibilité matérielle.
