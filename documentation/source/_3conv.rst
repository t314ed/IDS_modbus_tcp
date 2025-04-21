.. _3conv:

===================================
Conversion des models pour les SoC
===================================

Description
===========
Cette section traite de la conversion des modèles pour les systèmes sur puce (SoC).
Nous aborderons deux méthodes de conversion :

- Edge Impulse
- Emlearn

Ces outils permettent de convertir des modèles de machine learning pour une utilisation sur des microcontrôleurs et d'autres systèmes embarqués.
Nous obtiendrons des fichiers C/C++ : des librairies qui intègreront les modèles de machine learning.
Nous verrons réaliser les conversions.

Emlearn 
=========

Emlearn est un outil de conversion de modèles de machine learning pour les systèmes embarqués. 
Il permet de convertir des modèles en code C/C++ optimisé pour les microcontrôleurs.
Vous pouvez l'utiliser retrouver plus d'informations sur le site officiel d'Emlearn : https://emlearn.readthedocs.io/en/latest/
ou encore sur leur GitHub : https://github.com/emlearn
Le tutoriel d'installation est disponible sur le github.

Nous allons commencer par utiliser les bibliothèques suivantes :

.. code-block:: python

     import pickle
     import pandas as pd
     import emlearn


Nous avons la conversion suivante pour les modèles sauvés avec pickle :

.. code-block:: python

     #Lire le model
     rf_model = pickle.load(open(file_path+'random_forest.pkl', 'rb'))

     # Afficher les paramètres du modèle
     print(f"Type du modèle : {type(rf_model)}")
     print(rf_model.get_params())

     # Conversion 
     c_rf = emlearn.convert(rf_model, method='inline')
     c_rf.save(file=c_file_path+"rf_modbus.h", name="rf_modbus")

Il faut noter que votre modèle doit être pris en charge par Emlearn.

Une fois terminé, vous obtiendrez un librairie en C où se trouvera votre
model.

ML Studio / Edge Impulse
=========================

Edge impulse est une plateforme en ligne utilisé pour les entraînements de modèles. 
Elle cible particulièrement des cartes embarquées. ML Studio est basé sur Edge impulse,
mais sera plus axé sur les produits de Nordic. 

.. note::

     Lien de ML Studio : https://mlstudio.nordicsemi.com/

Sur cette plateforme nous pouvons faire tout notre entraînement. 
Mais nous choisirons de juste d'iporter notre model déjà entraîné.
En effet Edge Impulse offre la possibilité d'importé des modèles entrainnés
avec TensorFlow.

.. figure:: /_static/images/ml1.png

Arrivé à cette page, on choisi ``Upload your model``.

.. figure:: /_static/images/ml2.png

Sur cette page on choisira notre model développé avec tensor flow lite en ``.tflite``.
En suite on coche `Yes, run performance profiling for:` tout en choisissant bien notre carte.
Dans notre cas, on a utilisé la `Nordic nRF5340 DK`.

Laisser vous en suite guidé, et vous obtiendrez un ``.zip`` qui 
contiendra le model converti pour la carte.

Une documentation est disponible lors du téléchargement du fichier.

Vous trouverez un code pour l'utilisation de ce model.

