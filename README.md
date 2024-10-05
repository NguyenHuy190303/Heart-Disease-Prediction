# Heart Disease Prediction Using Machine Learning

This project explores multiple machine learning models to predict heart disease based on a given dataset. The notebook implements various classification algorithms and evaluates their performance.

## Dataset

The dataset contains various health metrics, which are used as features to predict whether a person has heart disease or not. The preprocessing steps handle missing values and ensure the data is ready for modeling.

## Models Used

The following models are implemented and evaluated for the prediction task:

- **K-Nearest Neighbors (KNN)**
- **Support Vector Machine (SVM)**
- **Naive Bayes**
- **Decision Tree**
- **Random Forest**
- **AdaBoost**
- **Gradient Boosting**
- **XGBoost**
- **Stacking (Ensemble Learning)**

## Preprocessing

- Handling missing data.
- Scaling features for models requiring normalized data.
- Splitting the dataset into training and test sets.

## Evaluation

Each model is evaluated based on accuracy for both the training and testing sets. Confusion matrices are generated to provide insights into the true positive, true negative, false positive, and false negative rates.

## Results

- Stacking was found to give the best performance with an accuracy of 92% on the training set and 90% on the test set.
  
## How to Run

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/HeartDiseasePrediction.git
    ```
2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the Jupyter notebook to train and evaluate the models:
    ```bash
    jupyter notebook ML_HeartDiseasePrediction.ipynb
    ```

## Requirements

- Python 3.7+
- Jupyter Notebook
- NumPy
- Pandas
- Scikit-learn
- XGBoost
- Matplotlib

## Conclusion

This project compares multiple machine learning algorithms for heart disease prediction, highlighting the power of ensemble methods like Stacking in improving prediction performance.
