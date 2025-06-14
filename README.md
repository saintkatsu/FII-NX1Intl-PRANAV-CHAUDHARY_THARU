# Cardiovascular Risk Assessment AI

A machine learning-based system that predicts cardiovascular disease risk using patient health metrics. The model provides risk assessment with confidence scores based on comprehensive health indicators.

## 🏥 Features

- Real-time cardiovascular risk prediction
- Confidence score for predictions
- Comprehensive health metrics analysis
- High-performance deep learning model

## 🛠️ Technology Stack

### Machine Learning & Data Processing
- **Python 3.x**
- **PyTorch**: Deep learning framework
- **scikit-learn**: Data preprocessing and model evaluation
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations

## 🤖 Machine Learning Model

### Architecture
The model uses a deep neural network with the following architecture:
- Input Layer: 14 features
- Hidden Layers:
  - Layer 1: 256 neurons with BatchNorm and Dropout (0.3)
  - Layer 2: 128 neurons with BatchNorm and Dropout (0.3)
  - Layer 3: 64 neurons with BatchNorm and Dropout (0.2)
  - Layer 4: 32 neurons with BatchNorm and Dropout (0.1)
- Output Layer: 2 neurons (Binary Classification)

### Features Used
1. Age (years)
2. Height (cm)
3. Weight (kg)
4. Systolic Blood Pressure
5. Diastolic Blood Pressure
6. Cholesterol Level
7. Glucose Level
8. Smoking Status
9. Alcohol Consumption
10. Physical Activity
11. BMI (Body Mass Index)
12. Blood Pressure Difference
13. Mean Arterial Pressure
14. Weight-Height Ratio

### Training Process
- **Cross-Validation**: 5-fold cross-validation
- **Optimizer**: AdamW with weight decay
- **Learning Rate**: Adaptive with ReduceLROnPlateau scheduler
- **Loss Function**: Cross Entropy Loss
- **Early Stopping**: Implemented to prevent overfitting
- **Batch Size**: 64
- **Data Preprocessing**: StandardScaler for feature normalization

### Model Performance
The model is evaluated using:
- Accuracy
- Precision
- Recall
- F1 Score

## 📊 Dataset
Link: https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset

The model is trained on the Cardiovascular Disease Dataset from Kaggle, which includes:
- 70,000 patient records
- 11 original features
- 3 derived features
- Binary classification labels


## 🙏 Acknowledgments
- Dataset provided by Kaggle
- Powered by PyTorch 