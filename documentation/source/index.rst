.. IDS_MODBUS documentation master file, created by
   sphinx-quickstart on Mon Mar 24 15:34:46 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. role:: raw-html(raw)
   :format: html

:raw-html:`<div style="float: right; margin-left: 20px; margin-top: -60px;">`

.. image:: _static/images/UBS-LOGO-RVB-Fd-Transparent.png
   :width: 200
   :align: right
   :alt: Logo UBS

:raw-html:`</div>`


Welcome to IDS_MODBUS TCP's documentation!
===========================================

Description
-----------
L'objectif est de fournir une vue d'ensemble sur les différentes étapes 
de développement et de déploiement d'un système de détection d'intrusion 
(IDS) pour le protocole Modbus TCP. 
Nous aborderons les aspects théoriques et pratiques, en mettant l'accent 
sur les techniques de détection d'anomalies et leur implémentation sur 
des systèmes sur puce (SoC) de type microcontrôleur et FPGA.

La cible de ce travail sera principalement la carte Nordic nRF5340 DK.

Pré-requis: 
-----------------------

- Carte Nordic nRF5340 DK
- Jupyter notebook
- Conda

.. note:: 
   
   Vous trouverez dans un fichier ``requirement1.txt`` les bibliothèques Python utilisé leurs versions.

Tables des matières :
----------------------
.. toctree::
   :maxdepth: 2

   _1model
   _2train
   _3conv
   _4dep_soc
   _5problem
   _FPGA

Indices and tables
------------------

* :ref:`search`
