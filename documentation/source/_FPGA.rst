.. _FPGA:

===============
Tutoriel : Déploiement d'un MLP sur FPGA avec PyTorch et Vitis AI
===============

Description
============
 
Ce tutoriel vous montre comment :  

1. **Entraîner un modèle MLP** avec PyTorch sur le dataset **SWAT** (cybersécurité industriel).  
2. **Optimiser le modèle** pour FPGA en utilisant **Vitis AI** (quantification, compilation pour DPU).  
3. **Déployer** le modèle sur une carte FPGA et mesurer **l'efficacité énergétique** durant l'inférence.  

**Points clés couverts :**  

- Configuration de Vitis AI et outils PyTorch.  
- Quantification pour des performances accrues sur DPU.  
- Benchmarking énergétique après déploiement.  

**Code & Guide Complet** : [GitHub Tutorial-MLP-with-PyTorch-on-SWAT-Dataset-with-Vitis-AI](https://github.com/IhebBenGaraAli/Tutorial-MLP-with-PyTorch-on-SWAT-Dataset-with-Vitis-AI)  


Comparaison de la consommation d'énergie entre un MLP avec FINN et un MLP avec Vitis AI sur un DPU
==================================================================================================

**Consommation du MLP avec Vitis AI sur DPU**  
- **Graphique de la consommation d'énergie :**  
  ![](https://codimd.math.cnrs.fr/uploads/upload_c941dd88631a7dd9054842183ae64aa4.png)  
  
.. figure:: https://codimd.math.cnrs.fr/uploads/upload_c941dd88631a7dd9054842183ae64aa4.png
   :alt: Graphique de consommation énergétique du MLP avec Vitis AI
   :align: center


- **Énergie totale consommée :** 952.56 Joules  
- **Puissance moyenne :** 5.15 W  

**Consommation du MLP avec FINN**  
- **Graphique de la consommation d'énergie :**  
![](https://codimd.math.cnrs.fr/uploads/upload_99099c50ce6141fbb3a84b17af02ee9f.png)

- **Énergie totale consommée :** 10 527.63 Joules  
- **Puissance moyenne :** 4.22 W  

**Analyse des résultats**  


1. **Avec FINN** :
   - **Énergie totale** : 10 527,63 Joules  
   - **Puissance moyenne** : 4,22 W  
   - **Temps d'exécution estimé** : `Énergie / Puissance = 10 527,63 / 4,22 ≈ 2 494 secondes` (~41,5 minutes)

2. **Avec Vitis AI (DPU)** :
   - **Énergie totale** : 751,92 Joules  
   - **Puissance moyenne** : 5,16 W  
   - **Temps d'exécution estimé** : `751,92 / 5,16 ≈ 145,7 secondes` (~2,4 minutes)

---

**Qui consomme le plus ?**
- **FINALEMENT, FINN consomme ~14x plus d'énergie** (10 527 J vs 751 J) que Vitis AI + DPU.  
- **Cependant, le DPU a une puissance moyenne légèrement supérieure** (5,16 W vs 4,22 W), ce qui signifie qu'il est plus agressif en calcul mais bien plus efficace.

---

**Analyse détaillée :**
**1. Pourquoi FINN consomme plus ?**
- **Approche basée sur FPGA pur** (sans accélérateur dédié comme le DPU) :
  - Moins optimisé pour les opérations MLP (réseaux entièrement connectés).
  - Peut nécessiter plus de cycles pour le même calcul.
- **Temps d'exécution bien plus long** (~41 min vs ~2,4 min), ce qui augmente l'énergie totale.

**2. Pourquoi Vitis AI + DPU est plus économe ?**
- **Le DPU (Deep Learning Processing Unit)** est un accélérateur matériel spécialisé :
  - Optimisé pour les inférences (quantification, parallélisme massif).
  - Réduit fortement le temps de calcul → économie d'énergie globale.
- **Meilleure efficacité énergétique** (Joules par opération).

---

**Comment interpréter la puissance moyenne ?**
- La **puissance (Watt)** indique le "taux" de consommation d'énergie à un instant donné.
- Le DPU a une **puissance moyenne plus élevée** (5,16 W vs 4,22 W), mais comme il termine bien plus vite, l'énergie totale est nettement inférieure.

→ **C'est comme comparer une voiture sportive (DPU) et un vélo (FINN)** :
- La voiture consomme plus de carburant **par minute**, mais elle fait le trajet en 2 minutes au lieu de 40 → économie globale.

---

**Conclusion : Vitis AI + DPU est bien plus efficace**
- **Énergie totale** : **Vitis AI gagne** (751 J << 10 527 J).  
- **Temps d'exécution** : **Vitis AI est plus rapide**.  
- **Efficacité énergétique** : Le DPU est optimisé pour le deep learning, contrairement à une implémentation FPGA générique (FINN).


