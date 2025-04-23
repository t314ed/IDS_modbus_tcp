.. _2train:

========================================
Développement des models : Entrainnement 
========================================

Description
===========
Pour les modèles, on utilisera les bibliothèques suivantes : 

- Tensorflow lite
- Scikit-learn
- Pytorch
- XGBoost 

Source utilisée pour réaliser le travail : https://github.com/SubrataMaji/IDS-UNSW-NB15/tree/master

Entrainement 
============

Nous suivons les étapes suivantes pour réaliser les entraînements :

- Charger les données
- Séparer les   les labels (Normal/Attack) et features(toutes autres colonnes)
- Paramètrer le modèle
- Entraîner le modèle
- Tester et évaluer le modèle
- Sauvegarder le modèle

Les paramètrages des modèles ont été pris selon les résultats des travaux IDS-UNSW-NB15.
On peut noter que d'autres paramètres peuvent être utilisés pour d'autres modèles. 
Nous avons entrainé les modèles suivantes avec leurs paramètres respectifs :

Modèles supervisés 
------------------

1. Pytorch

Turotiel suivi : https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html

    -  MLP (Multi Layer Perceptron)
    On utilisera les paramètres suivants pour ce modèle : 

            - 3 couches 
            - 64 neurones par couche
            - 10 epochs

    .. code-block:: python

        class MLP(nn.Module): 
            def __init__(self, input_size):
                super(MLP, self).__init__()
                self.l1 = nn.Linear(input_size, 64) #couche 1 , input_size = nb features
                self.l2 = nn.Linear(64,64)          #couche 2
                self.l3 = nn.Linear(64,1)           #couche 3
                self.sigmoid = nn.Sigmoid()

            def forward(self, x):
                x = torch.relu(self.l1(x))  # Activation ReLU après la 1ère couche
                x = torch.relu(self.l2(x))  # Activation ReLU après la 2ème couche
                x = self.sigmoid(self.l3(x))  # Activation Sigmoid pour la sortie
                return x

    Nous devons réaliser des manipulations sur les données pour les transformer en tenseurs PyTorch.

    .. code-block:: python 

        # manipulation data
        batch_size = 1000
        
        # Standardisation des données
        scaler = StandardScaler()
        x_train_scal = scaler.fit_transform(x_train)
        x_test_scal  = scaler.fit_transform(x_test)
        
        # Convertir les données en tensor pytorch
        x_train_tensor = torch.tensor(x_train_scal, dtype = torch.float32)
        y_train_tensor  = torch.tensor(y_train, dtype = torch.float32)
        x_test_tensor  = torch.tensor(x_test_scal, dtype = torch.float32)
        y_test_tensor  = torch.tensor(y_test, dtype = torch.float32)
        
        # Créer les Dataloader pour les entrainnement et test
        train_data = TensorDataset(x_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_data, batch_size = batch_size, shuffle=True)
        
        test_data = TensorDataset(x_test_tensor, y_test_tensor)
        test_loader = DataLoader(test_data, batch_size = batch_size, shuffle=True)
                    
    On peut instanscier le modèle comme suit :

    .. code-block:: python

        nb_features = 46 # nombre de features
        mlp = MLP(nb_features) # Instancier le modèle

    L'entrainnement se fera de cette façon :

    .. code-block:: python

        num_epochs = 10 # nombre d'epochs
        # Définir la fonction de perte et l'optimiseur
        criterion = nn.BCELoss()  # Perte binaire pour classification binaire
        optimizer = optim.Adam(mlp.parameters(), lr=0.001)

        for epoch in range(num_epochs):
            mlp.train()
            running_loss = 0.0

            for inputs, labels in train_loader:
                # Zero gradients
                optimizer.zero_grad()

                # Passage avant
                outputs = mlp(inputs)

                # Calcul de la perte
                loss = criterion(outputs.squeeze(), labels.float())

                # Backpropagation
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
        print("____ Training Succeesded")     


    Pour évaluer ce modèle, on peut utiliser le code suivant :

        .. code-block:: python

            mlp.eval()
            with torch.no_grad():
                y_pred = mlp(x_test_tensor).squeeze()
                y_pred_class = (y_pred > 0.5).long()  # Si la probabilité > 0.5, prédire 1, sinon 0

                accuracy = (y_pred_class == y_test_tensor).float().mean()
                print(f"Précision MLP: {accuracy:.5f}")

        Dans le notebook, nous avons une fonction qui nous donne une évaluation plus avancée du modèle: "evaluate_result_torch()". 

        .. code-block:: python

            evaluate_result_torch(my_model, x_features_train , y_label_train, x_features_test, y_label_test, model_name='my_model' , scaler=scaler)
        
        Cette fonction nous donne les résultats suivants :
            - Tableau de score pour l'entraînement et le test : précision, F1-score et Fausses alertes. 
            - Courbe de précision
            - Matrice de confusion

        Pour des utilisations, on peut récupérer ces données de test qui sont retournées par la fonction: 
         nom du model, précision, f1 score et fausses alertes.

        Exemple d'utilisation : 

        .. code-block:: python

            # Chargement des données sans les transformer en tenseurs
            x_train, y_train = train_data.drop(columns=['Normal/Attack']), train_data['Normal/Attack']

            scaler = StandardScaler()
            # Récupérer les résultats
            model_name, auccuracy, f1_score, false_alerts = evaluate_result_torch(mlp, x_train, y_train, x_test, y_test, model_name='MLP 3-64' , scaler=scaler)
            
            # Afficher les résultats
            print(f"Nom du modèle: {model_name}")
            print(f"Précision: {accuracy:.5f}")
            print(f"F1-score: {f1_score:.5f}")
            print(f"Fausses alertes: {false_alerts:.5f}")

         

        Pour sauvegarder le modèle, on peut utiliser la commande suivante :

        .. code-block:: python

            file_path = 'models/'  # Chemin où vous souhaitez enregistrer le modèle
            torch.save(mlp.state_dict(), file_path+'mlp_model.pth')
            
        
    2. Scikit-learn

        - MLP (Multi Layer Perceptron) 

            Ce MLP aura les mêmes paramètres que le MLP de Pytorch.
            Nous allons instancier notre modèle de la façon suivante :

            .. code-block:: python

                mlp_sk = MLPClassifier(
                    hidden_layer_sizes=(64, 64, 64),  
                    activation='relu',           # ReLU comme fonction d'activation
                    solver='adam',               # Optimiseur Adam
                    max_iter=500,                # Nombre maximal d'itérations
                    random_state=42
                )
            
            Pour l'entrainement, on peut utiliser la commande suivante :

            .. code-block:: python

                mlp_sk.fit(x_train, y_train)
             
            Nous disposons de plusieurs moyen pour effectuer l'évaluation de ce modèle.

            .. code-block:: python

                # Réaliser les prédictions
                y_pred = mlp_sk.predict(x_test)
                accuracy = accuracy_score(y_test, y_pred)
                print(f"Précision MLP : {accuracy:.5f}")
                
                # Utilisation de fonction de score
                score = mlp_sk.score(x_test, y_test)
                print(f"Précision MLP : {score:.5f}")

            Nous disposons aussi de la fonction "evaluate_result()" qui a inspiré la fonction "evaluate_result_torch()".
            Nous retrouverons donc les mêmes propriétés, mais cette fonction est modélisé pour les modèles de Scikit-learn.
            Voici sa syntaxe :

            .. code-block:: python

                evaluate_result(my_model, x_features_train, y_label_train, x_features_test, y_label_test, 'Model name')

            Cette fonction nous donne les résultats suivants :
                - Tableau de score pour l'entraînement et le test : précision, F1-score et Fausses alertes. 
                - Courbe de précision
                - Matrice de confusion

            Exemple d'utilisation :

            .. code-block:: python

                model_name, accuracy, f1_score, false_alerts = evaluate_result(mlp_sk, x_train, y_train, x_test, y_test, 'MLP with sklearn')

                # Afficher les résultats
                print(f"Nom du modèle: {model_name}")
                print(f"Précision: {accuracy:.5f}")
                print(f"F1-score: {f1_score:.5f}")
                print(f"Fausses alertes: {false_alerts:.5f}")

            Cette fonction sera fonctionnelle pour les autres modèles de Scikit-learn ci-dessous.
            Sauvegarde du modèle :

            .. code-block:: python

                file_path = 'models/'  # Chemin où vous souhaitez enregistrer le modèle
                pickle.dump(mlp_sk, open(file_path+'mlp_sk.pkl', 'wb'))


        - Random Forest

        Nous avons choisi d'utiliser une foncton "GridSearchCV" pour optimiser les paramètres de ce modèle.

        .. code-block:: python

            rf = RandomForestClassifier()
            param = {
                'n_estimators': [10, 15, 20],
                'max_depth': [5, 10, 15]
            }

            gds = GridSearchCV(estimator=rf, param_grid=param, cv=5, scoring='accuracy', n_jobs=-1)
            
            # Entraînement du modèle
            gds.fit(x_train, y_train)

            # Meilleurs paramètres
            print("Meilleurs paramètres :", gds.best_params_)
            best_rf = gds.best_estimator_

        On notera que les paramètres utilisés dans param peuvent être modifiés. Dans ce travail, ils ont été minimisé pour des 
        raison de performance de la machine utilisée.

        Pour l'évaluation, on peut utiliser la fonction "evaluate_result()" comme pour le MLP de Scikit-learn.
        Ou encore, on peut utiliser la commande vu précédement avec le MLP de Scikit-learn.
        Pour sauvegarder le modèle, on peut utiliser la commande qu'on a vu avec 'pickle'.


        - Decision Tree


        Nous restons sur la même démarche que précédement, 
        c'est-à-dire que nous allons utiliser la fonction "GridSearchCV" pour optimiser les paramètres de ce modèle.

        .. code-block:: python

            dt = DecisionTreeClassifier()

            param = {'criterion': ['gini', 'entropy'],
                     'max_depth':[8, 10, 12, 14],
                     'min_samples_split':[2, 4, 6],
                     'min_samples_leaf': [1, 2, 5]
            }

            gds = GridSearchCV(estimator=dt, param_grid=param, cv=5, scoring='accuracy', n_jobs=-1)
            gds.fit(x_train, y_train)
            # Meilleurs paramètres
            print("Meilleurs paramètres :", gds.best_params_)

            #prendre le meilleur modèle
            best_dt = gds.best_estimator_


        Notez que vous pouvez modifier les paramètres 'param' pour explorer d'autres combinaisons. 
        Ceux là ont été choisis pour des raisons de performance de la machine utilisée, donc assez réduits.

        Aucun changement, nous pouvons utiliser la fonction "evaluate_result()" pour évaluer le modèle.
        Ou les méthodes d'évaluation vues précédement avec le MLP de Scikit-learn.
        La sauvegarde du modèle se fait de la même manière que pour le MLP de Scikit-learn.


        - Logistic Regression

        Nous allons instancier le modèle de la manière suivante :

        .. code-block:: python

            lr_model = SGDClassifier(penalty='l1', alpha=1e-6)

        On peut en suite l'entraîner :

        .. code-block:: python

            lr_model.fit(x_train, y_train)

        Aucun changement, nous pouvons utiliser la fonction "evaluate_result()" pour évaluer le modèle.
        Ou les méthodes d'évaluation vues précédement avec le MLP de Scikit-learn.
        La sauvegarde du modèle se fait de la même manière que pour le MLP de Scikit-learn.
        

        - SVM (Support Vector Machine)

            1.  Linear SVC(Support Vector Classifier)

                Si nous voulons utiliser SGDClassifier avec une optimisation de GridSearchCV, nous pouvons le faire de la manière suivante :

                .. code-block:: python

                    linear_svc = SGDClassifier(loss='hinge')
                    # hyperparam_tuning
                    param = {'alpha':[10**x for x in range(-5,3)],  # Values for alpha
                             'penalty':['l1', 'l2']} 
                    cv=3

                    custom_scorer = make_scorer(fbeta_score, beta=2)

                    tuning_clf = GridSearchCV(linear_svc, param, scoring=custom_scorer, refit='auc',
                                                      cv=cv, verbose=3, return_train_score=True)

                    linear_svc.fit(x_train, y_train)

                D'un autre côté nous pouvons aussi instancier ce modèle directement avec LinearSVC de Scikit-learn :

                .. code-block:: python

                    linear_svc = LinearSVC(C=1.0, max_iter=1000)

                

            2. SVC (Support Vector Classifier)

            Nous allons implementer ce modèle de la manière suivante :  

            .. code-block:: python

                svc_ = SVC(kernel='rbf', C=1.0, gamma='scale', probability=False, shrinking=True)

        Pour ces deux modèles, l'entrainement, l'évaluation et la sauvegarde se font de la même manière que pour le MLP de Scikit-learn.

    3. XGBoost

        - XGBoost Classifier

        Pour instancier ce modèle, nous avons les paramètres suivants :

        .. code-block:: python

            best_params = {'n_estimators':400,
               'max_depth':12,
               'learning_rate':0.1,
               'colsample_bylevel':0.5,
               'subsample':0.1,
               'n_jobs':-1}

        Ces paramètres ont été déterminés par les travaux de l'IDS-UNSW-NB15. Pour des raisons de performance, nous
        avons réduit les paramètres.

        .. code-block:: python

            best_params = {'n_estimators':40,
               'max_depth':10,
               'learning_rate':0.1,
               'colsample_bylevel':0.5,
               'subsample':0.1,
               'n_jobs':-1}

        On peut instancier le modèle comme suit :

        .. code-block:: python

            xgb_clf = xgb.XGBClassifier(**best_params)


        Puis, on l'entraîne :

        .. code-block:: python

            xgb_clf.fit(x_train, y_train)

        On peut évaluer le modèle de la même manière que les autres modèles de Scikit-learn: avec la fonction "evaluate_result()".
        La préduiction se fait comme suit : 

        .. code-block:: python

            y_pred = xgb_clf.predict(x_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Précision XGBoost : {accuracy:.5f}")

        On peut maintenant sauvegarder le modèle :

        .. code-block:: python

            file_path = 'models/'  # Chemin où vous souhaitez enregistrer le modèle
            pickle.dump(xgb_clf, open(file_path+'xgb_model.pkl', 'wb'))

        On peut noter que dans nos travaux, la précision de ce modèle est de 0.99432 .
        Ce qui est très bon, malgré le fait que les paramètres soient réduits.

    4. Tensorflow lite

        - MLP (Multi Layer Perceptron) 

        Ce modèle est similaire à celui de Pytorch et Scikit-learn.
        Nous allons l'instancier de la manière suivante :

        .. code-block:: python

            model = tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation='relu', input_shape=(x_train.shape[1],)),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(1, activation='sigmoid') # classification binaire
            ])
            model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
        
        Nous allons l'entraîner de la manière suivante :

        .. code-block:: python

            model.fit(x_train, y_train, epochs=10)

        Nous pouvons évaluer le modèle de la manière suivante :

        .. code-block:: python

            # Évaluation du modèle
            loss, accuracy = model.evaluate(x_test, y_test)
            print(f'Test Accuracy: {accuracy:.5f} , Loss : {loss:.5f}')

        On va la sauvegarder de la manière suivante :

        .. code-block:: python

            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            tflite_model = converter.convert()

            # Sauvegarder le modèle TensorFlow Lite
            with open('ML3_64.tflite', 'wb') as f:
                f.write(tflite_model)

        Ce fichier sera utilisé pour la conversion.

Modèles non supervisés
-------------------------

    - Clustering KMeans 

    Pour ce modèle, nous n'avons pas besoin de préciser les labels, car il s'agit d'un modèle non supervisé.
    Nous allons donc l'instancier de la manière suivante :

    .. code-block:: python

        cluster = KMeans(
            init ="random", 
            n_clusters =2, # 2 clusters pour Normal et Attack
            n_init=10,
            max_iter=300,
            random_state=42
        )  

    Nous allons l'entraîner de la manière suivante :

    .. code-block:: python

        # Standardisation des données
        scaler = StandardScaler()
        x_train_clust_scal = scaler.fit_transform(x_train)
    
        # Entraînement du modèle
        cluster.fit(x_train_clust_scal)

    Nous pouvons évaluer le modèle de la manière suivante :

    .. code-block:: python

        # Récupérer les labels de test
        y_test_clust = test_data_clust['Normal/Attack']

        # Prédictions des clusters sur les données de test
        y_test_pred = cluster.predict(x_test_clust_scal)

        # Mapper les clusters aux labels réels
        mapping = {}
        for cluster_label in np.unique(y_test_pred):  # Parcourir les clusters prédits
            # Sélectionner les vrais labels correspondant aux points du cluster
            true_labels = y_test_clust[y_test_pred == cluster_label]  # Indexation booléenne NumPy

            if len(true_labels) > 0:
                # Assigner le label majoritaire du cluster
                cluster_mode = mode(true_labels, nan_policy='omit')
                # Si mode est un scalaire, il n'est pas nécessaire d'indexer
                mapping[cluster_label] = cluster_mode.mode
            else:
                mapping[cluster_label] = None

        # Vérifier si le mapping est valide
        if None in mapping.values():
            raise ValueError("Certains clusters n'ont pas pu être associés à des labels réels.")

        # Appliquer le mapping aux prédictions
        y_test_pred_mapped = [mapping[cluster] for cluster in y_test_pred]

        # Calculer l'accuracy
        accuracy = accuracy_score(y_test_clust, y_test_pred_mapped)

        print(f"Accuracy: {accuracy:.5f}")



Documentations Python : 
---------------------

- PyTorch: https://pytorch.org/docs/
- Scikit-learn: https://scikit-learn.org/stable/documentation.html
- XGBoost: https://xgboost.readthedocs.io/
- TensorFlow Lite: https://www.tensorflow.org/lite/guide





