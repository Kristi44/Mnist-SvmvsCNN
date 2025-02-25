   ```markdown
   # Classification des chiffres manuscrits avec SVM et CNN

   Ce projet utilise le dataset MNIST pour classifier des chiffres manuscrits à l'aide de SVM (Support Vector Machine) et CNN (Convolutional Neural Network).

   ## Installation

   1. Clonez ce dépôt :
      ```bash
      git clone https://github.com/VotreNomUtilisateur/NomDuDepot.git
      cd NomDuDepot
      ```

   2. Installez les dépendances :
      ```bash
      pip install numpy matplotlib tensorflow scikit-learn
      ```

   ## Utilisation

   ### 1. Importer les bibliothèques et le dataset
   ```python
   import matplotlib.pyplot as plt
   import numpy as np
   import tensorflow as tf

   mnist = tf.keras.datasets.mnist 
   (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
   ```

   ### 2. Prétraitement des données
   - Réduire la taille du dataset de 30 fois
   - Filtrer les classes 0, 1 et 2
   - Vectoriser et normaliser les données
   ```python
   num_sample = X_train.shape[0] // 30
   indices = np.random.choice(X_train.shape[0], num_sample, replace=False)
   X_train_reduced = X_train[indices]
   Y_train_reduced = Y_train[indices]

   mask = np.isin(Y_train_reduced, [0, 1, 2])
   X_train_filtred = X_train_reduced[mask]
   Y_train_filtred = Y_train_reduced[mask]

   X_train_vectorized = X_train_filtred.reshape(X_train_filtred.shape[0], -1).astype("float32")
   from sklearn.preprocessing import StandardScaler
   scaler = StandardScaler()
   X_train_norm = scaler.fit_transform(X_train_vectorized)
   ```

   ### 3. Visualisation des données
   ```python
   from sklearn.decomposition import PCA
   pca = PCA(n_components=2)
   X_train_pca = pca.fit_transform(X_train_norm)

   plt.figure(figsize=(10, 8))
   scatter = plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=Y_train_filtred, cmap="viridis", alpha=0.7)
   plt.colorbar(scatter, ticks=[0, 1, 2], label="Classes")
   plt.show()
   ```

   ### 4. Classification avec SVM
   ```python
   from sklearn.svm import SVC
   clf_gauss = SVC(kernel="rbf", gamma="scale", C=1.0)
   clf_gauss.fit(X_train_norm, Y_train_filtred)

   # Évaluation
   from sklearn.metrics import accuracy_score
   Y_pred_test = clf_gauss.predict(X_test_norm)
   accuracy_test = accuracy_score(Y_test_filtred, Y_pred_test)
   print(f"Exactitude : {accuracy_test:.5f}")
   ```

   ### 5. Matrice de confusion
   ```python
   from sklearn.metrics import confusion_matrix
   conf_matrix = confusion_matrix(Y_test_filtred, Y_pred_test)
   plt.figure(figsize=(10, 7))
   plt.imshow(conf_matrix, interpolation='nearest', cmap='Blues')
   plt.title('Matrice de confusion')
   plt.colorbar()
   plt.ylabel('Vraies étiquettes')
   plt.xlabel('Prédictions')
   plt.show()
   ```

   ### 6. Classification avec CNN
   ```python
   X_train_cnn = X_train_filtred[..., tf.newaxis]
   X_test_cnn = X_test_filtred[..., tf.newaxis]

   X_train_cnn_norm = X_train_cnn / 255.0
   X_test_cnn_norm = X_test_cnn / 255.0

   model = tf.keras.models.Sequential([
       tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
       tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
       tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
       tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
       tf.keras.layers.Flatten(),
       tf.keras.layers.Dense(128, activation='relu'),
       tf.keras.layers.Dense(10, activation='softmax')
   ])

   model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
   model.fit(X_train_cnn_norm, Y_train_filtred, epochs=10, validation_data=(X_test_cnn_norm, Y_test_filtred))

   test_loss, test_acc = model.evaluate(X_test_cnn_norm, Y_test_filtred)
   print(f"Précision pour les données de test : {test_acc:.5f}")
   ```

   ## Résultats
   Les résultats montrent que le modèle CNN atteint une précision de test plus élevée par rapport au modèle SVM, ce qui suggère une meilleure performance pour la classification des chiffres manuscrits.

   ## Contributeurs
   - **VotreNom** - Développeur principal

   ## Licence
   Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.
   ```
