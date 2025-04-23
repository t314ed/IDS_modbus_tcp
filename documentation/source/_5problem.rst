.. _5problem:


Résultats obtenus
==========================


Emlearn
---------

- Emprunte mémoire :

Lors du build, on obtient l'emprunte mémoire qui sera utilisé pour notre programme :

.. code-block:: shell

    Memory region         Used Size  Region Size  %age Used
           FLASH:       78144 B         1 MB      7.45%
             RAM:       10624 B       448 KB      2.32%
        IDT_LIST:          0 GB        32 KB      0.00%


- Temps pris (nombre de tick d'horloge) pour une prédiction :

.. code-block:: shell

 One prediction : 4647644446702174208  , 0
 One prediction : 4647714815446351872  , 1
 One prediction : 4647679631074263040  , 2
 One prediction : 4647609262330085376  , 3
 One prediction : 4647626854516129792  , 4
 One prediction : 4647591670144040960  , 5
 One prediction : 4647591670144040960  , 6
 One prediction : 4647732407632396288  , 7
 One prediction : 4647609262330085376  , 8
 One prediction : 4647609262330085376  , 9
 One prediction : 4647609262330085376  , 10
 One prediction : 4647450932655685632  , 11
 One prediction : 4647433340469641216  , 12
 One prediction : 4647433340469641216  , 13
 One prediction : 4647433340469641216  , 14
 One prediction : 4647433340469641216  , 15
 One prediction : 4647433340469641216  , 16
 One prediction : 4647433340469641216  , 17
 One prediction : 4647433340469641216  , 18
 One prediction : 4647433340469641216  , 19


- Consommation d'énergie: 

.. figure:: /_static/images/emlearn1.png

.. figure:: /_static/images/emlearn2.png

On peut voir ici que 200 prédictions prennent 4s. 
On peut aussi voir qu'on est aux environs de 3 mA pour la consommation, soit 2.4 - 3.7 mA.

ML Studio
-----------

- Emprunte mémoire : 

.. code-block:: shell

    Memory region         Used Size  Region Size  %age Used
           FLASH:      141364 B         1 MB     13.48%
             RAM:       19944 B       448 KB      4.35%
        IDT_LIST:          0 GB        32 KB      0.00%

- Temps pris (nombre de tick d'horloge) pour une prédiction : 

.. code-block:: shell

    features 18
    Prediction here
        Attack: 0.28783
        Normal: 0.71217
    expected prediction: 1.000000 vs  prediction 
    Time for one prediction : 4630404104378646528 ticks


- Consommation d'énergie : 


.. figure:: /_static/images/mlenergy1.png

.. figure:: /_static/images/mlenergy2.png

On peut observer une  consommation dans les environs de 10 mA soit un interval de 7 - 12 mA.
On notera que nous effectuons 10 000 prédictions.
On peut aussi voir que pour ces prédictions prennent environ 12s.



Problèmes rencontrés
========================

Problème pour effectuer un flash sur la carte :
-------------------------------------------------

Nous avons rencontré des problèmes lors de l'écriture sur la carte. 
Voici la sortie du terminal : 

.. code-block:: shell

    west flash -d /home/telly/Documents/anomalie_detection/lab/blinky_pwm/build --dev-id 1050039768 --erase

    -- west flash: rebuilding
    [0/5] Performing build step for 'blinky_pwm'
    ninja: no work to do.
    [2/5] No install step for 'blinky_pwm'
    [3/5] Completed 'blinky_pwm'
    [4/5] cd /home/telly/Documents/anomalie_detection/lab/blinky_pwm/build/_sysbuild && /home/telly/ncs/toolchains/b81a7cd864/usr/local/bin/cmake -E true
    -- west flash: using runner nrfjprog
    -- runners.nrfjprog: mass erase requested
    -- runners.nrfjprog: reset after flashing requested
    -- runners.nrfjprog: Flashing file: /home/telly/Documents/anomalie_detection/lab/blinky_pwm/build/merged.hex
    [error] [ Client] - Encountered error -5: Command select_coprocessor executed for 1 milliseconds with result -5
    ERROR: Failed when selecting coprocessor APPLICATION
    [error] [ Worker] - Encountered unexpected debug port ID 0, expected 6
    ERROR: The --family option given with the command (or the default from
    ERROR: nrfjprog.ini) does not match the device connected.
    NOTE: For additional output, try running again with logging enabled (--log).
    NOTE: Any generated log error messages will be displayed.
    FATAL ERROR: command exited with status 18: nrfjprog --program /home/telly/Documents/anomalie_detection/lab/blinky_pwm/build/merged.hex --chiperase --verify -f NRF53 --coprocessor CP_APPLICATION --snr 1050039768

     *  The terminal process terminated with exit code: 18. 
     *  Terminal will be reused by tasks, press any key to close it. "

Ce problème a été rencontré lors de l'utilisation du module de mesure de consommation.

La solution est de mettre un $jumper$ sur les pins P22.

.. figure:: /_static/images/jumper.jpeg

Par la suite, on va exécuter les commandes suivantes pour restaurer la carte : 

.. code-block:: shell

    $  nrfjprog --coprocessor CP_NETWORK --recover -c 4 && nrfjprog --recover -c 4

        Recovering device. This operation might take 30s.
        Erasing user code and UICR flash areas.
        Writing image to disable ap protect.
        Recovering device. This operation might take 30s.
        Erasing user code and UICR flash areas.
        Writing image to disable ap protect.


Si ces commmandes ne marchent pas, on pourrait essayer  les suivantes : 

.. code-block:: shell

    $ nrfjprog --coprocessor CP_NETWORK --recover -f NRF53
        Recovering device. This operation might take 30s.
        Erasing user code and UICR flash areas.
        Writing image to disable ap protect.
    
    $ nrfjprog --recover -f NRF53
        Recovering device. This operation might take 30s.
        Erasing user code and UICR flash areas.
        Writing image to disable ap protect.



.. note::

    Il faut noter qu'il est préférable de maintenir ce jumper en place si on n'exécute aucune mesure de courant.

    Des précautions ont été énoncé dans ce guide pour la préparation à la mesure de courant pour la carte : 
    https://docs.nordicsemi.com/bundle/ug_nrf5340_dk/page/UG/dk/prepare_board.html