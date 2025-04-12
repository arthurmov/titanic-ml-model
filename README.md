# titanic-ml-model
### Titanic Survival Prediction

This project applies machine learning techniques to predict passenger survival on the Titanic using the [Kaggle Titanic dataset](https://www.kaggle.com/competitions/titanic/data). 

It was developed as part of the CISC 3440 Machine Learning course.

---

### Project Overview

The goal is to classify whether a passenger survived or not based on key features such as class, age, sex, fare, and more. Two main models are implemented:

- **Decision Tree Classifier**: For interpretability and performance.
- **Data Preprocessing & Feature Engineering**: Includes encoding categorical variables and binning numerical features.

---

### Dataset

The dataset is available on Kaggle:

[Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic/data)

> **Note**: Due to licensing restrictions, the dataset (`train.csv`) is **not included** in this repository.  
To run the project:
1. Download `train.csv` from the Kaggle link above.
2. Place it in the project root directory.

---

### Features Used

| Feature      | Description                                      |
|--------------|--------------------------------------------------|
| `Pclass`     | Ticket class (1st, 2nd, 3rd)                     |
| `Sex`        | Gender of the passenger                          |
| `Age`        | Age in years                                     |
| `SibSp`      | # of siblings/spouses aboard                     |
| `Parch`      | # of parents/children aboard                     |
| `Fare`       | Ticket price                                     |
| `Embarked`   | Port of embarkation (C, Q, S)                    |

---

### Installation & Requirements

Install required Python libraries using:

```bash
pip install numpy pandas scikit-learn matplotlib
