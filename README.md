# What 2 Play? — Recommender Project

What 2 Play? is a project that focuses on allowing a user to get Boardgame recommendations based on popularity, but that can also be updated with personal preferences, allowing the app to be personalized for each user.

This project integrates database analysis with **TF-IDF vectorization** and **cosine similarity** in a hybrid model, to rank and recommend relevant boardgames based on user preferences.

---
---
## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [Key Features](#-key-features)
3. [Project Structure](#-project-structure)
4. [Dataset](#-dataset)
5. [Notebooks](#-notebooks)
6. [Technology Stack](#-technology-stack)
7. [Environement](#-environement)
8. [License](#-license)

---
---
## 🔍 Project Overview

What2Play consists of **two core functionalities**:

1. **🔩 ML model for Personalized Recommendations**
   - Uses different models to generate **boardgame recommendations** based on user choices in specific features using **TF-IDF & cosine similarity**.
   - Allows users to select **favourite games** for more refined recommendations.

2. **✍🏻 Creation of user database**
   - Retrieves **user ratings** from the app.
   - **Merges** the preferences database with the main one.
   - Ensures users get **meaningful** boardgame recommendations based on his preferences.

### 🤩 All presented in a user friendly Streamlit app 🤩

---
---

## 🃏 Key Features:

### 🔍 Wide Database Search  
- Uses trained models to suggest **specific recommendations** tailored to user inputs.  
-  Users can select **predefined categories**, such as:
    - 🫂 **Number of Players**
    - ⏱️ **Duration of the game**
    - 🧠 **Complexity**
    - ⌛ **Age of the game**

### 🎯 Personalized Recommendations  
- Uses **cosine similarity** to recommend games based on user favourites
    - Multiple games can be enter as input

### 🔄 Constant Update
- Gets **user feedback** to update recommendations

### 💻 Streamlit App
- Prepared to store the preferences of different users
---
---
## 📂 Project Structure

```
boardgame-recommender/
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Feature_Engineering.ipynb
│   ├── 03_Model_Training.ipynb
│   └── 04_Recommender_System.ipynb
├── data/
│   ├── raw
│   └── processed
│   		├── boardgames_clean.csv
│   		└──boardgames_features.csv
├── models/
│   └── boardgame_rating_predictor.pkl
├── app/
│   └── streamlit_app.py
├── environement.yml
├── W2P_presentation.pdf
└── README.md
```
---
---
## 📦 Dataset

Two datasets combined with ~ 30.000 games and more than 50 columns of details were used

***Source:*** 
- The details of each game are from [BoardGameGeek Reviews on Kaggle](https://www.kaggle.com/datasets/jvanelteren/boardgamegeek-reviews/data)
- The most updated ranking file from [@beefsack bgg-ranking-historicals](https://github.com/beefsack/bgg-ranking-historicals)

⚠️ Because the size of the data is bigger than GitHub limits, the .csv files used can be found [**here**](https://drive.google.com/drive/folders/18iq5IuQWWr86HXXdAHXHER4fmTguAhOF?usp=sharing)

---
---
## 📓 Notebooks

1. **Exploratory Data Analysis**
    - Insights about the data used to work
2. **Feature Engineering**
    - Transformation of the data to be able to qork with it
3. **Model Trainig**
    - Different aproaches to get the best performing model
4. **Recommender System**
    - Combining the model and the similarity system to get the working code to be used on the Streamlit app

---
---
## 🛠 Technology Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.x |
| Training Models | Linear Regression, Random Forest, Gradient Boosting, XGB, CatBoost, ExtraTres, Ridge  |
| Hyperparameters | GridSearchCV |
| Extraction & Features | TF-IDF, TruncatedSVD, Kmeans |
| Data & Utilities | NumPy, pandas |
| Visualization | Matplotlib, Seaborn |
| Evaluation | scikit-learn (classification report, confusion matrix) |

---
---
## ⚙️ Environement

The environement to make this project was created using conda.
- To recreate the environment:

    `conda env create -f environment.yml -n project-env`

- To activate it:

    `conda activate project-env`

---
---
## 🛠️ Next Steps
- [x] Update the database
- [ ] Link to YouTube "How to Play" videos
- [ ] Publish it
---
---
## 📜 License

This project is licensed under the **MIT License**.
