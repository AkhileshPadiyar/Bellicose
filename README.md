#  Disaster Management
## Pipelining
1. Data Preprocessing
   - Removed Baseic
     ```
      def preprocess_img(img):
      img = cv2.GaussianBlur(img, (5,5), 0)
  
      lab = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2LAB)
      l,a,b = cv2.split(lab)
  
      clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
      l = clahe.apply(l)
  
      img = cv2.merge((l,a,b))
      img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
  
      img = img / 255.0
      return img
     ```
2. Model Selection
   - ResNet. But Why ?
     - Prevent vanishing gradient problems in very deep models.
     - Skip connections let information flow directly across layers.
     - ResNet enables building networks with hundreds or even thousands of layers.
     - It is widely used in computer vision tasks like image classification and object detection

3. Innovation
   - Instead of oversampling we used class_weights, to solve the problems of unbaised datasets
   - This reduces the time required to train the model, because of prevention of oversampling
     ```
      from sklearn.utils.class_weight import compute_class_weight
      class_weights = compute_class_weight(
      class_weight='balanced',
      classes=np.unique(train_data.classes),
      y=train_data.classes
     ```
     and Implemented in the compilation phase
     ```
       history = model.fit(
      train_data,
      validation_data=val_data,
      epochs=10,
      class_weight=class_weights
)
     ```
)

class_weights = dict(enumerate(class_weights))
     ```

4. Model Evaluation
   - Acheived an **trainng accuracy** and **validation accuracy** of **0.9402** and  **0.8758**
   - Losses were **loss: 0.1531** and  **val_loss: 0.4040**

##  Refrences
- [TensorFlow ResNet](https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet)
- [Github Pages for Documentation Writing Syntax](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax)
- [Keras](https://keras.io/api/applications/resnet/)
