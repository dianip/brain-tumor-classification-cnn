# Brain Tumour Classification with CNN

This project utilises a Convolutional Neural Network (CNN) to classify binary images of MRI brain scans, distinguishing between patients with and without brain tumours.

---

## Project Structure

- `brain_tumour_cnn.ipynb`: Main notebook containing preprocessing, model training, evaluation, and single-image testing.
- `data/`: Folder containing MRI images — split into `yes`, `no`, and `test image` directories.
- `model/brain-tumor-classification-cnn_model.keras`: Saved trained CNN model in Keras format.
- `slides//Brain Tumour Detection with CNN.pdf`: Contains presentation slides summarising the project in PDF format (`Brain Tumour Detection with CNN.pdf`).


---

## Tools & Libraries

- Python (Jupyter Notebook)
- tensorflow, numpy
- opencv-python, matplotlib
- scikit-learn, Pillow

---

## What’s Inside

The notebook includes:

- **Data Preprocessing**: Loads and resizes MRI images to 64x64, normalises pixel values, and assigns binary labels.
- **Train/Validation/Test Split**: Splits the dataset into 3 sets for model training and evaluation.
- **CNN Model Architecture**:
  - 3 convolutional layers with ReLU activation
  - 2 max-pooling layers
  - Flatten layer followed by:
    - 1 dense layer (64 units, ReLU)
    - 1 output layer (2 units)
- **Model Training**: Trains the model over 10 epochs using Adam optimiser and sparse categorical crossentropy.
- **Performance Evaluation**: Plots training vs validation accuracy and evaluates test set performance.
- **Single Image Prediction**: Tests the trained model on a single unseen MRI scan for tumour detection.

---

## Insights

- Final training accuracy reached **83.3%**.
- Validation accuracy peaked at **79.6%**, ending at **77.6%** by the final epoch.
- Test accuracy also achieved **77.6%**, suggesting solid generalisation but room for optimisation.
- A single unseen test image was correctly classified as **Tumour (Predicted Class: 1)**.

---

## Full Presentation Slides  
[View Slides (PDF)](slides/Brain_Tumour_Detection_with_CNN.pdf)

---

## Model Notebook  
The full notebook used for image loading, model training, and evaluation:  
[View Notebook](brain_tumour_cnn.ipynb)

---

## Dataset

This project uses the [**Brain MRI Images for Brain Tumour Detection**](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection) dataset from Kaggle, created by Navoneel Chakrabarty (accessed on 15 July 2025).

---

## Model Output  
The trained CNN model is saved here:  
[Download Model (.keras)](model/brain-tumor-classification-cnn_model.keras)
