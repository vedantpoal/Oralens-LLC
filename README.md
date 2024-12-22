# Oral Disease Classification Using Deep Learning

## Project Overview
This project focuses on classifying oral diseases, specifically Caries and Gingivitis, using deep learning techniques. The implemented model employs a Convolutional Neural Network (CNN) to predict the disease present in an image with high accuracy and robustness to variations in lighting, quality, and orientation.

---

## Features
- **Multiclass Classification:** Classifies each image into one of two categories: `Caries` or `Gingivitis`.
- **Evaluation Metrics:** Provides accuracy, precision, recall, F1-score, and confusion matrix for model evaluation.
- **Single Image Prediction:** Allows the user to input a single image and receive a prediction with confidence.
- **Visualization:** Plots training history and confusion matrix for performance analysis.

---

## Tech Stack
- **Programming Language:** Python
- **Libraries and Frameworks:**
  - TensorFlow/Keras
  - OpenCV
  - NumPy
  - Matplotlib
  - Seaborn
  - Scikit-learn

---

## Dataset
- The dataset is organized into `train` and `test` directories, each containing subdirectories for `Caries` and `Gingivitis` images.
- **Preprocessing Steps:**
  - Images are resized to 224x224 pixels.
  - Pixel values are normalized to the range [0, 1].

---

## Model Architecture
- **Input:** \(224 \times 224 \times 3\) RGB images
- **Layers:**
  - Convolutional layers with ReLU activation
  - MaxPooling layers for dimensionality reduction
  - Dropout layers for regularization
  - Fully connected dense layers
- **Output:** Two classes (Caries and Gingivitis) with softmax activation

---

## How to Run
### Prerequisites
- Python 3.7+
- Install required libraries:
  ```bash
  pip install tensorflow opencv-python matplotlib seaborn scikit-learn
  ```

### Steps
1. **Prepare Dataset:**
   - Organize the dataset as:
     ```
     dataset/
     |-- train/
     |   |-- Caries/
     |   |-- Gingivitis/
     |-- test/
         |-- Caries/
         |-- Gingivitis/
     ```

2. **Train the Model:**
   - Update dataset paths in the script.
   - Run the script to train the model:
     ```bash
     python oral_disease_classification.py --train
     ```

3. **Evaluate the Model:**
   - Evaluate the model on the test set:
     ```bash
     python oral_disease_classification.py --evaluate
     ```

4. **Predict a Single Image:**
   - Provide an image path and run:
     ```bash
     python oral_disease_classification.py --predict "path/to/image.jpg"
     ```

---

## Evaluation Metrics
- **Accuracy:** Measures overall correctness.
- **Precision:** Proportion of true positive predictions.
- **Recall:** Proportion of actual positives identified correctly.
- **F1-Score:** Harmonic mean of precision and recall.
- **Confusion Matrix:** Visual representation of prediction performance.

---

## Results
- Accuracy: 87%
- Precision: 88%
- Recall: 87%
- F1-Score: 87%

---

## Visualizations
- **Training History:**
  - Accuracy vs. Epochs
  - Loss vs. Epochs
- **Confusion Matrix:** Annotated heatmap showing prediction accuracy per class.

---

## Limitations
- Limited dataset size may affect model generalization.
- Model performance may drop on unseen data with significant variations.

---

## Future Enhancements
- Data augmentation to improve robustness.
- Use of transfer learning with pre-trained models like ResNet or EfficientNet.
- Expand dataset to include more disease categories.

---

## License
This project is open-source and available under the MIT License.

---

## Contact
For any queries or contributions, feel free to contact:
- **Name:** Vedant Poal
- **Email:** vedantpoal@gmail.com

---

## Acknowledgments
- TensorFlow and Keras for deep learning support.
- OpenCV for image processing.
- The contributors and maintainers of open-source libraries used in this project.
