# Treue-technology

## Bank Churn Prediction Model
This repository contains a machine learning model for predicting bank customer churn. Customer churn refers to the phenomenon where customers stop doing business with a company—in this case, leaving their bank. The model aims to identify potential churners so that the bank can take proactive measures to retain them.

![Customer-Churn](https://github.com/imshubhamkore/Treue-technology-Bank-Churn/assets/128685230/95c73d8a-25ad-4355-94cd-50829d9c1350)


Table of Contents
Introduction
Dataset
Dependencies
Usage
Model Training
Model Evaluation
Results
Contributing


### Introduction
Customer churn is a critical issue in the banking industry. This project provides a predictive model that can help banks identify customers who are likely to churn. By identifying potential churners early, the bank can take targeted actions to retain these customers, thereby improving customer retention rates.

### Dataset
The dataset used for this project can be found in the data directory. It includes historical customer data with features such as customer age, account balance, transaction history, and more. The target variable is Churn, which indicates whether a customer has churned (1) or not (0).

### Dependencies
To run the code in this repository, you will need the following dependencies:
Python 3.x
Google colab
Jupyter Notebook (optional but recommended for exploring the code)
Pandas
NumPy
Scikit-Learn
Matplotlib (for data visualization)
Seaborn (for data visualization)

### Usage
To use the bank churn prediction model, follow these steps:

Clone this repository to your local machine:
bash
git clone https://github.com/imshubhamkore/Treue-technology/blob/db1756a71d818bf362acc428a77d1e104c7ae0de/Bank_Customer_Churn.ipynb
cd bank-churn-prediction

Install the required dependencies as mentioned in the "Dependencies" section.
Open and explore the Google colab Bank_Churn_Prediction.ipynb for a detailed walkthrough of the model, including data preprocessing, model training, and evaluation.
You can also use the pre-trained model in your own Python scripts or applications by loading it from the models directory.

### Model Training
The model training process is documented in the Jupyter Notebook Bank_Churn_Prediction.ipynb. It covers data preprocessing, feature engineering, model selection, and SMOTE.

### Model Evaluation
We evaluate the model's performance using metrics such as accuracy, precision, recall, and F1-score. The evaluation results are discussed in the Google Colab.

# Machine Learning Models Performance

This repository contains the performance metrics of various machine learning models evaluated on a specific dataset. The table below summarizes the precision, recall, F1-score, and accuracy for each model.

| **Model**               | **Precision** | **Recall** | **F1-Score** | **Accuracy** |
|-------------------------|---------------|------------|--------------|--------------|
| Logistic Regression     | 0.57          | 0.54       | 0.52         | 0.59         |
| Gradient Boosting       | 0.70          | 0.69       | 0.69         | 0.71         |
| Random Forest           | 0.70          | 0.69       | 0.69         | 0.70         |


### Results
The results of the churn prediction model can be summarized here. You can also include visualizations and insights gained from the analysis.

### Contributing
Contributions to this project are welcome! If you have suggestions or improvements, please open an issue or submit a pull request. For major changes, please discuss them in advance.
