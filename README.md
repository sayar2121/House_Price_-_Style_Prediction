# HomeValue AI 🤖

HomeValue AI is a comprehensive web application that leverages machine learning to provide **instant, data-driven predictions** for house prices and architectural styles. This project demonstrates a full-stack data science application—from **data processing and model training** to a **responsive, interactive web interface**. The models are trained on the **Ames Housing dataset** from Kaggle, ensuring robust and realistic predictions.

> (Note: You can take a screenshot of your running application, upload it to a site like Imgur, and replace the URL below to include an image in your README.)

---
## DATASET Link- https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques

## ✨ Features

- **Dual Predictions**:  
  - **Regression**: Random Forest Regressor predicts the continuous value of a home's sale price.  
  - **Classification**: Random Forest Classifier identifies the property's architectural style (e.g., 1-Story, 2-Story).  

- **Interactive & Responsive UI**:  
  Built with a mobile-first design, ensuring seamless experience on desktops, tablets, and smartphones.

- **Explainable AI (XAI)**:  
  Includes a dynamic **feature importance chart**, showing which inputs impacted the final price prediction the most.

- **Full-Stack Architecture**:  
  - **Backend**: Flask serves the pre-trained models via REST API.  
  - **Frontend**: Vanilla HTML, CSS, and JavaScript handle all user interaction and visualization.

---

## 🛠️ Tech Stack

- **Backend**: Python, Flask  
- **Machine Learning**: Scikit-learn, Pandas  
- **Frontend**: HTML5, CSS3, JavaScript  
- **Data Visualization**: Chart.js for dynamic feature importance charts  

---

## 📂 Project Structure

HOUSE_PREDICTION_MODEL/
```
├── app/
│ ├── house_price_model.pkl # Regression model for price prediction
│ └── house_style_model.pkl # Classification model for style prediction
├── templates/
│ └── index.html # Main HTML file served by Flask
├── venv/ # Python virtual environment (not included in repo)
├── backend.py # Flask server & API
├── train_all_models.py # Script to preprocess data and train models
└── requirements.txt # Python dependencies
---

## 🚀 Local Setup and Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
## 🚀 Local Setup and Installation
```
### 2. Download The Dataset
```bash
# 1. Visit the Ames Housing Kaggle page
# 2. Download 'train.csv'
# 3. Create a folder named 'data' in the project root and move 'train.csv' into it
mkdir data
# Move train.csv into the data folder manually or using your file explorer
```
### 3. Create and Activate Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate

```
### 4. Create and Activate Virtual Environment
```bash
pip install -r requirements.txt
 
✅ This way, the **dataset instructions** and **virtual environment setup** are all inside readable code blocks on GitHub.  

If you want, I can now **update your full README** with this style included so it’s ready to paste directly into GitHub. Do you want me to do that?

```
## ⚙️ Usage
1. Train The Model
```bash
python train_all_models.py
```
This processes the raw data, trains both the regression and classification models, and saves them as .pkl files in app/.
2. Run the Backend Server
```bash
python backend.py
```
3. Access the Application
- Open your browser and navigate to: http://127.0.0.1:8080
- Input home features and get real-time predictions for price and architectural style.
