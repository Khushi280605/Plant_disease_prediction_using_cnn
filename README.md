# 🌿 Plant Disease Detection Using Convolutional Neural Networks (CNN)

## 📌 Overview

This project implements a deep learning–based Plant Disease Detection system using Convolutional Neural Networks (CNN).  
The model is trained on the **PlantVillage dataset (~88,000 images)** across **38 crop–disease classes** and deployed using **Streamlit** for real-time leaf image classification.

The system predicts plant diseases from uploaded leaf images and supports sustainable agriculture through automated diagnosis.



## 🚀 Features

- Multi-class classification (38 crop-disease categories)
- Custom CNN architecture (10 Convolution layers)
- Dropout regularization to reduce overfitting
- Adam optimizer with categorical crossentropy loss
- Evaluation using Accuracy, Precision, Recall, F1-Score
- Confusion Matrix visualization
- Real-time deployment using Streamlit

  
## 🔄 Prediction Workflow

1. User uploads leaf image  
2. Image resized to 128×128  
3. Converted to RGB format  
4. Converted to NumPy array  
5. Passed into trained CNN model  
6. Softmax probabilities generated  
7. Argmax selects predicted class  
8. Disease name displayed  
## 📚 Libraries Used

- TensorFlow / Keras
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- OpenCV
- Streamlit
- Pillow


## 🗂 Dataset Information

- **Dataset:** PlantVillage
- **Total Images:** ~88,000
- **Classes:** 38 crop–disease categories
- **Validation Samples:** 17,572 images
- **Image Size:** 128 × 128 pixels
- **Color Mode:** RGB (3 Channels)



## 🧠 Model Architecture

The CNN architecture consists of:

- 5 Convolution Blocks  
  - Conv2D → ReLU → Conv2D → ReLU → MaxPooling  
- Dropout (0.25)
- Flatten Layer
- Dense Layer (1500 neurons, ReLU)
- Dropout (0.4)
- Output Layer (38 neurons, Softmax)

### Architecture Summary

| Layer Type        | Count |
|-------------------|--------|
| Convolution Layers| 10     |
| Pooling Layers    | 5      |
| Dropout Layers    | 2      |
| Flatten Layers    | 1      |
| Dense Layers      | 2      |



## ⚙ Training Configuration

- **Optimizer:** Adam (learning rate = 0.0001)
- **Loss Function:** Categorical Crossentropy
- **Batch Size:** 32
- **Epochs:** 10
- **Hidden Activation:** ReLU
- **Output Activation:** Softmax



## 📊 Model Performance

| Metric | Value |
|--------|--------|
| Training Accuracy | ~98% |
| Validation Accuracy | ~96.5% |
| F1-Score | 0.97 |
| Classes | 38 |

Evaluation Metrics Used:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix



## 📁 Project Structure

```bash
Plant-Disease-Detection/
│
├── train_plant_disease.ipynb
├── test_plant_disease.ipynb
├── main.py
├── trained_plant_disease_model.keras
├── training_hist.json
├── Diseases.png
└── README.md
```



## 🛠 Installation

### 1️⃣ Clone Repository

```bash
git clone https://github.com/Khushi280605/Plant_disease_prediction_using_cnn.git
cd Plant_disease_prediction_using_cnn
```

### 2️⃣ Install Dependencies

```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn streamlit pillow opencv-python
```



## ▶ Training the Model

Open and run:

```bash
train_plant_disease.ipynb
```

This notebook:
- Loads dataset
- Trains CNN
- Evaluates performance
- Saves model as `.keras`



## 🧪 Testing the Model

Run:

```bash
test_plant_disease.ipynb
```

This will:
- Load trained model
- Predict disease for test image
- Display predicted class



## 🌐 Running the Streamlit App

```bash
streamlit run main.py
```

Then open in browser:

Upload a leaf image to get real-time disease prediction.





## ⚠ Limitations

The model is trained on PlantVillage dataset with controlled backgrounds.  
Real-world images with background noise and lighting variations may reduce prediction accuracy due to domain shift.



## 🔮 Future Improvements

- Apply stronger data augmentation
- Use transfer learning (MobileNet / EfficientNet)
- Fine-tune on real-world field images
- Add disease treatment suggestions
- Deploy on cloud (AWS / Render / Azure)







## ⭐ Conclusion

This project demonstrates the effectiveness of Convolutional Neural Networks for automated plant disease classification.
