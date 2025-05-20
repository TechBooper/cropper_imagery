
# 🖼️ Cropper Imagery — Outil de découpe intelligent d'images par détection de visages (en construction)

Un projet personnel : **Cropper Imagery** permet de **découper automatiquement des images** à partir de la détection de visages. L’idée ? Gagner du temps dans des tâches répétitives comme :

- Préparer des photos de profil
- Formater des visuels pour les réseaux sociaux
- Produire en lot des images prêtes pour le e-commerce

C’est un outil en Python, pensé pour être rapide, modulaire, et adaptable.

---

## ⚙️ Fonctionnalités

- 📸 Détection de visages + points de repère
- 📐 Découpe selon différents formats (Instagram, LinkedIn, TikTok…)
- 🔄 Correction automatique de rotation
- 🖼️ Traitement par lot d’un dossier entier
- 🧪 Aperçu rapide avant traitement
- 🪞 Affinage possible (netteté, filtres, marges…)

---

## 📁 Structure du projet

cropper_imagery/
│
├── cropper/                 # Modules de découpe selon parties du visage
│   ├── crop_chin_image.py
│   ├── crop_nose_image.py
│   └── ...
├── processing.py            # Script principal pour traitement par lot
├── gradio_app.py            # Interface test basique avec Gradio
├── presets.json             # Formats de découpe disponibles
├── README.md

---

## 💡 Utilisation rapide

### 1. Installation

git clone https://github.com/TechBooper/cropper_imagery.git
cd cropper_imagery
pip install -r requirements.txt

### 2. Traitement d’un dossier

Modifier les chemins dans processing.py, puis lancer :

python processing.py

### 3. Aperçu de la découpe

Pour tester un seul fichier avec aperçu (Gradio requis) :

python gradio_app.py

---

## 🔧 Formats inclus (presets)

- instagram_square → ratio 1:1
- linkedin_cover → ratio 1.91:1
- tiktok_story → ratio 9:16
- headbust → découpe centrée sur le haut du visage
- 🎯 100% personnalisable via presets.json

---

## 🙋‍♂️ Pourquoi ce projet ?

Je m'appelle **Marwane Wafik**, développeur Python junior basé en Île-de-France. Ce projet a été conçu comme un exercice technique, avec des **cas d’usage réels**.

🔧 J’utilise Python pour automatiser, structurer et proposer des solutions simples à des problèmes précis.
🎯 Objectif court terme : rejoindre une équipe tech (freelance, CDI, CDD, alternance ou stage accepté).

---

## 📩 Contact

- GitHub : TechBooper
- Email : marwanewafik@gmail.com *(à adapter si besoin)*
- Localisation : Gennevilliers (92), disponible immédiatement

---

## 📝 À venir

- Interface utilisateur complète (Tkinter ou web)
- Amélioration du système de filtre
- Intégration CLI plus avancée
- Tests unitaires

---

## 📄 Licence

MIT — projet libre et réutilisable
