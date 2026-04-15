Detailed Preprocessing Steps
- Loading and Converting (cv2.imread,cvtColor(BGR2RGB))In this images are loaded and converted from BGR to RGB, as it is standard for most CNNs model, ensuring colors are represented correctly.
- Resizing (cv2.resize(img,(224,224))): Standardizes all images to 224x224 pixels. CNNs require uniform input sizes, and 224x224 is a standard input size for many pre-trained models.
- Noise Reduction (cv2.GaussianBlur): Smooths the image to reduce, unwanted noise, which helps the model focus on structure rather than pixel-level artifacts.
- Contrast Enhancement (CLAHE): The image is converted to LAB color space, and Contrast Limited Adaptive Histogram Equalization (CLAHE) is applied. This improves local contrast, highlighting important details without over-enhancing noise.
- Normalization (img / 255.0): Scales pixel values from 


. This helps the neural network converge faster during training.

- Label Encoding (LabelEncoder): Converts string labels into numeric vectors for model compatibility.
- Class Balancing (resample): Detects class imbalance and uses resample to oversample minority classes. This ensures the model does not become biased toward the majority class, improving accuracy on underrepresented classes.
- One-Hot Encoding (to_categorical): Converts labels into categorical format, which is required for multi-class classification output layers usually softmax activation function.
- Data Augmentation (ImageDataGenerator): Applies transformations like rotation, shifting, zooming, and flipping . This artificially increases the dataset size, reduces overfitting, and improves model generalization.


