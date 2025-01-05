# Aerogel-Bonding-290241
The aim of the project is to understand whether the aerogel bonding is good enough for the commercial application. 

## Team Members
- *Francesco Costa 299271*
- *Lorenzo Cinelli *
- *Sebastian Jr Foumane*

---

## 1. Introduction

This project focuses on analyzing the **aerogel bonding** process for commercial applications. The dataset (`aerogel_bonding.csv`) contains various worker-related and process-related features, including performance metrics, job history, and material handling statistics. The ultimate goal is to predict whether each aerogel bonding instance is **successful** (`BondingSuccessful = 1`) or **unsuccessful** (`BondingSuccessful = 0`).

This is a **binary classification** task, as the target variable is strictly 0 or 1.

---

## 2. Exploratory Data Analysis (EDA)

### 2.1 Dataset Overview
  First of all we imported all the necessary Python Libraries including: pandas,numpy, matplotlib.pyplot and seaborn.
  We loaded the dataset and inspected its size, columns, and data types using:
  ```python
  df.shape, df.info(), df.head()
  ```
  This was useful for us to have a clearer overview of the dataset provided.
  We found that the dataset is composed by 20000 rows and 31 columns (27 numerical columns and 4 categorical columns).
  We summarized numeric columns and identified potential outliers or skewed distributions with `df.describe()`.
  We visualized missing values.
  We realized that several columns have missing values and also with the help of a heatmap that showed a large amounts of them we started 
  to make some considerations that will take part during the preprocessing phase (which alternative to adopt if imputation or removal).

### 2.2 Data Visualization
For this phase we took the help of a lot of graphycal element to have a clearer represenation of some particular variables.
- **Histograms/Boxplots**:
  Thanks to the histogram of the distribution of all the numerical variables we understood that the majority of the numercial features 
  exhibit skewed distribution meaning there is a possible requirement of normalization or transformations. Analyzed distributions of features like `BondingRiskRating`, `MistakesLastYear`, etc.
  The distribution of categorical variables too made us look at their imbalanced situation.
  The distribution of target variable instead was useful to understand that BondingSuccessful is a very unbalanced variable with more         unsuccessful than successful.
-  Analyzed distributions of features like `BondingRiskRating`, `MistakesLastYear`, etc.
- **Countplots for Categorical Variables**: Explored columns like `JobStatus` and `CivilStatus`.
- **Correlation Heatmap**: Identified potential feature relationships (e.g., `SkillRating` vs. `WorkExperience`).
  The tool of the Correlation heatmap was essential to understand relationships between all the numerical variables.
  Thanks to it we decided to drop some of the 27 numerical columns that were useless for our analysis for redundancy reason or because were not related to our target variable.

### 2.3 Observations
With the boxplot for the numerical variables we noticed the presence of lots of outliers
- Certain columns exhibited **extreme outliers** (e.g., `ProcessedKilograms`), requiring removal or capping.
  Removal of outlier could be a crucial part of the next phase of our project , the preprocessing, but as we can show further ahead the models two of three models that we use for our analysis are very robust to outliers and our intention is to not reduce drastically the dataset even because we removed already some columns and we knwo that the models we use are very efficient with large dataset. 
- **Missing values** in some columns necessitated imputation strategies.
- The target variable (`BondingSuccessful`) displayed some level of imbalance (validated using `df['BondingSuccessful'].value_counts()`).

---

## 3. Methods

### 3.1 Environment Setup
- **Python version**: 3.x  
- **Libraries**:
  - `pandas`, `numpy`: Data manipulation
  - `matplotlib`, `seaborn`: Data visualization
  - `scikit-learn`: Machine learning models and utilities
  - `missingno` (optional): Visualizing missing data
  - `imblearn` (optional): Handling imbalanced datasets (e.g., SMOTE)

Install all dependencies via:
```bash
pip install -r requirements.txt
```

### 3.2 Data Preprocessing

#### Outlier Removal
- Used statistical methods (e.g., IQR or z-scores) to remove extreme values in columns such as `RequestedProcessAmount`, `ProcessedKilograms`, and `BondingRiskRating`.

#### Handling Missing Values
- Assessed missingness using:
  ```python
  df.isnull().sum()
  ```
- Applied strategies like:
  - Dropping columns with excessive missing data.
  - Imputing moderate missingness with mean/median or using `KNNImputer`.

#### Encoding Categorical Features
- Transformed columns like `JobStatus`, `HighestEducationAttained`, and `CivilStatus` using `pd.get_dummies` or `OneHotEncoder`.
- The target variable (`BondingSuccessful`) was already numeric and required no additional processing.

#### Scaling
- Applied `StandardScaler` to numerical features (e.g., `ProcessedKilograms`, `BondingRiskRating`), particularly for models like Logistic Regression or KNN.

#### Train-Test Split + Validation
- Split the dataset into training (80%) and test (20%) sets.
- Further split the training data into training/validation sets or applied cross-validation during model tuning.

---

## 4. Model Selection & Hyperparameter Tuning

### 4.1 Chosen Models
We tested the following models:

1. **Logistic Regression**: A simple and interpretable baseline.
2. **Random Forest**: An ensemble method effective for tabular data.
3. **Gradient Boosting**: A boosting algorithm that incrementally improves predictions.


### 4.2 Hyperparameter Tuning
- **Logistic Regression**:
  - Tuned `C` (regularization strength) and `solver` (`liblinear`, `lbfgs`, etc.).
- **Random Forest**:
  - Tuned `n_estimators` (number of trees), `max_depth`, and `min_samples_split`.
- **Gradient Boosting**:
  - Tuned `learning_rate`, `n_estimators`, and `max_depth`.

Used `GridSearchCV` or manual loops to optimize hyperparameters based on validation accuracy and F1-score.

### 4.3 Metrics
- **Accuracy**: Proportion of correct predictions.
- **F1-score**: Harmonic mean of precision and recall (useful for imbalanced datasets).
- **ROC-AUC**: Evaluates model performance across classification thresholds.

---

## 5. Results

### Summary of Model Performance (Example Metrics):

| Model               | Accuracy | F1 Score | Notes                                              |
|---------------------|----------|----------|---------------------------------------------------|
| Logistic Regression | 0.90     | 0.88     | Baseline model; used `C=10`, `solver='lbfgs'`.    |
| Random Forest       | 0.93     | 0.92     | Performed well with `n_estimators=200`.          |
| Gradient Boosting   | 0.94     | 0.93     | Best results with `learning_rate=0.1`, `max_depth=5`. |

### Key Observations
- Logistic Regression provided an explainable baseline.
- Ensemble methods (Random Forest, Gradient Boosting) captured nonlinear patterns and performed better.
- Gradient Boosting showed the best overall performance.

(Optional): Include visualizations like confusion matrices, ROC curves, or classification reports to explain the models' performance.

---

## 6. Conclusions

### 6.1 Main Findings
- Gradient Boosting was the most effective model for predicting aerogel bonding success.
- Key features influencing predictions included `BondingRiskRating`, `RequestedProcessAmount`, `MistakesLastYear`, and `SkillRating`.

### 6.2 Limitations
- Some columns (e.g., `ProcessingTimestamp`, `ApplicantAge`) had limited predictive value and might introduce noise.
- Imbalanced datasets (if applicable) may require oversampling or class weighting for better performance.

### 6.3 Future Work
- **Feature Engineering**: Develop new features, e.g., the ratio of mistakes to total tasks.
- **Expand Dataset**: Larger datasets may improve model robustness.
- **Deployment**: Integrate the best-performing model into a monitoring pipeline for real-time predictions.

---

## 7. Reproducibility

### Environment
- **Python version**: 3.x
- **Key Packages**: pandas, numpy, scikit-learn, matplotlib, seaborn, etc.

### How to Run
1. Clone this repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the notebook or script:
   ```bash
   python main.py
   ```

---

## Thank You

For any inquiries or suggestions, feel free to contact the project authors. We hope this project serves as a comprehensive example of building a binary classification pipeline for aerogel bonding success predictions.

