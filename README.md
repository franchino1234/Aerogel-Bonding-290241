# **Aerogel Bonding**

### **Team Members**
- Francesco Costa 299271
- Lorenzo Cinelli 292481
- Sebastian Jr. Foumane 290241 (“captain”)

---

## **Introduction**

### **Project Overview**
The **Aerogel Bonding Project** focuses on analyzing the bonding performance of aerogels with various materials to evaluate its commercial potential. Aerogels, known for their exceptional thermal insulation and lightweight properties, hold significant promise for industrial applications. However, their success depends heavily on bonding quality, particularly in:
- **Aerospace structures**
- **Industrial machinery**
- **Solid-state batteries**

### **Our Aim**
This project aims to:
1. Determine whether the current aerogel bonding process is commercially viable.
2. Identify key factors that influence bonding success using data-driven methods.
3. Provide actionable recommendations to optimize the process, reduce risks, and improve performance.

The analysis is applied to the **aerogel_bonding.csv** dataset, which captures details about workers, process parameters, and outcomes, with a target variable called `BondingSuccessful` indicating success or failure (0 or 1).

---

## **Methods**

The methodology for this project is designed to ensure a comprehensive analysis of the aerogel bonding process.

### **Data Understanding**

We loaded the dataset and found that it consists of 20,000 rows and 31 columns. These columns include several variables that provide information related to the **Aerogel Bonding process** and the **Success of Bonding**.  
The features are of three types:
- **Numerical features** such as `BondingRiskRating` and `TotalMaterialProcessed`.
- **Categorical features** such as `CivilStatus` and `JobStatus`.
- **Date features** such as `ProcessingTimeStamp`.

### **Exploratory Data Analysis (EDA)**

We imported the necessary libraries, including: Pandas, Numpy, Matplotlib.pyplot, and Seaborn.

 **Feature Analysis**:

   After obtaining an overview of the basic information (which is useful for our reasoning and analysis) of our dataset, we started the Feature Analysis.  
   First, we focused on the presence of missing values and realized that they were abundant.  
   As we can see from the first two variables `HolidaysTaken` with 1905 and `PercentageOfCompletedTasks` with 2045, going down the list, we noticed that many other features show a significant number of missing values.  
   Checking missing values is crucial in data analysis and preprocessing for several reasons, such as data quality, model performance, preprocessing decisions, and feature evaluation.  
   To better visualize their distribution, we created a heatmap that shows, through a light cell, a massive concentration of missing values, confirming what we saw in the list of missing values.

   ![Heatmap of missing values](output.png)




   We then focused on the target variable to observe its behavior and distribution, knowing that it is a binary variable (`1` for success, `0` for failure).  
   We decided to plot its distribution and found that it was highly imbalanced.

   ![Distribution of bonding succesful](bondsucces.png)

   As we can see from the image, the unsuccessful case (`0`) with 13,690 instances is much larger than the successful case (`1`) with 4,307.  
   This imbalance needed to be addressed.

### **Balancing the Dataset**

We performed "Balancing" to address the class imbalance in the dataset, ensuring that all classes in the target variable were equally represented. This is crucial for:
- Preventing model bias
- Improving accuracy
- Enhancing generalization

How did we apply it?

We focused on identifying and addressing missing data in unsuccessful bonding cases to improve data quality. By prioritizing rows with the highest number of missing values, we could effectively clean, impute, or remove problematic data points, ensuring more accurate and unbiased model training.  
We analyzed unsuccessful bonding rows (`BondingSuccessful = False`). By sorting these rows based on the number of missing values, we prioritized cleaning or removal of problematic data. This was very helpful for ensuring that our dataset was reliable and minimized the risk of introducing noise into the model.  
Since the number of unsuccessful bonding rows exceeded the successful ones, we reduced the excess by removing the top 9,383 "worst offenders" rows with the highest number of missing values. This ensured that both classes (successful and unsuccessful bonding) were equally represented, which was critical for preventing bias during model training.  
With the balanced dataset, we enhanced the model's ability to learn equally from both classes, improving its accuracy and generalization. Additionally, removing rows with excessive missing data reduced computational overhead and strengthened data quality. The final dataset was shuffled to randomize its distribution, making it more robust for machine learning algorithms. The distribution was now balanced, with 4,307 for success and 4,307 for failure. The previously added column to contain missing values and the sorting of rows in descending order were deleted as they were no longer necessary.

### **After Balancing**

Now that we had balanced our dataset, we continued analyzing the behavior of all our variables and searched for particular relationships between them.

We saw that, even after balancing, except for the target variable (which had 0 missing values), the other variables still had missing values, so further preprocessing was needed before modeling.  
However, there was a positive change: the heatmap after balancing showed missing values distributed rather than concentrated.

#### **Data Visualization and Analysis of Single Variables**

To understand the behavior of each specific variable, we plotted histograms.

We used them to understand the distribution of numerical features, identify skewness or outliers that may need handling, and assess data readiness for modeling by spotting patterns or irregularities. We realized that in the dataset, many features, like `RequestedProcessAmount` and `ProcessedKilograms`, were right-skewed, indicating most values were low with a few outliers. Others, like `Dependability` and `Trustability`, followed a normal distribution. Some features, such as `HolidaysTaken`, displayed discrete patterns, while `MistakesLastYear` was heavily imbalanced. `BondingSuccessful` was obviously concentrated between 0 and 1.  
These observations suggested the need for preprocessing steps like normalization, outlier handling, and addressing class imbalances to improve data quality and model performance.

   ![Numerical data distribution](datadistr.png)

We did the same work for the categorical features and found that they were imbalanced, with some categories (e.g., `Bachelor`, `Employed`, `Married`) dominating the dataset. These could introduce bias into machine learning models.

#### **Correlation Matrix**

Another important analysis was the correlation between numerical features.  
We identified relationships and dependencies between numerical features in the dataset. Features with high correlation (positive or negative) told us about redundancy or strong influence, guiding feature selection and model design.  
After plotting the matrix, we realized that:
- A strong correlation (0.98) was found between `TotalMaterialProcessed` and `OtherCompaniesMaterialProcessed`.
- `ChurnRisk` and `TotalChurnRisk` showed redundancy, with `TotalChurnRisk` often excluded due to overlap with `SkillRating`.
- `ApplicantAge` might add noise if it didn’t improve predictions, while `WorkHistoryDuration` and `ProcessingTimestamp` appeared irrelevant to bonding success and could be removed to simplify the model.

Thus, we decided to drop some columns to reduce our dataset size and make it more manageable for analysis:
- `OtherCompaniesMaterialProcessed`
- `SkillRating`
- `WorkHistoryDuration`
- `TotalChurnRisk`
- `ApplicantAge`
- `ProcessingTimestamp`

We conducted further analysis, examining the relationships between the target variable and selected features.  
We specifically wanted to highlight the pairplot of the most important variables for the target variable.

   ![Pairplot of most important variables for target](pairplot.png)

This analysis uncovered valuable insights that could guide improvements prior to commercializing the bonding process.

### **Addressing Missing Values and Outliers**

Before, we only removed missing values for `BondingSuccessful`, but we had to do it for the other variables as well.  
During this phase, we focused on identifying missing values, outliers, and feature relationships to make informed preprocessing decisions.

- **Missing Values**:  
To assess missing data, we calculated their total counts and visualized the results from the heatmap shown earlier, which revealed that missing values were evenly distributed across the dataset. For imputation, we tailored our approach to the data type:
  - **Numerical Variables**: We opted for the median as an imputation strategy because it handles skewed distributions well and is robust to outliers. Many features in our dataset displayed skewness, making this method ideal.
  - **Categorical Variables**: The most frequent value was chosen, as it preserved the original distribution of the variable.  
  These methods offered an effective balance between simplicity and performance, avoiding the added complexity of alternatives like KNN-imputation, which could increase computational demands, reduce interpretability, and risk overfitting.

- **Outliers**:  
Outliers in numerical features were detected using the BoxPlot method. Box plots provided a visual summary of key statistics, including the median, interquartile range, and potential **outliers**. They were particularly useful for identifying outliers, understanding variability, and comparing features.

   ![Boxplot shows outliers](ouliers.png)

As the boxplot showed, many of the variables showed significant outliers.  
Upon further consideration, we concluded that removing outliers from all features would have led to the loss of valuable data, as anomalies often carried critical information in industrial datasets.  
Instead, we opted to retain all outliers and relied on machine learning models that are robust to outliers. This approach minimized preprocessing efforts, avoided potential bias, and preserved the dataset size, which was already reduced by prior feature selection. 

### **Scaling and Encoding**

- **Scaling**:  
To standardize feature ranges, we applied "StandardScaler" via a pipeline. This approach was well-suited for our dataset, which included features with both long ranges (e.g., `ProcessedKilograms`) and short ranges (e.g., `HolidaysTaken`). By ensuring a mean of 0 and a standard deviation of 1 for all features, scaling made them comparable and improved performance for models sensitive to feature magnitudes, such as Logistic Regression.

- **Encoding**:  
Categorical variables were converted into numerical format using "OneHotEncoder". This technique created a binary column for each unique category in a feature, assigning a `1` to the column corresponding to the observed category in a row and `0` to all others. This method avoided assuming any ordinal relationship between categories, ensuring, for instance, that `Employed`, `Self-Employed`, and `Unemployed` were treated as independent and unordered groups. After encoding, the dataset gained 12 additional columns.  
These steps ensured the dataset was properly prepared for modeling, with consistent scaling and accurate representation of categorical data.

---

## **Models**

Given the binary target variable `BondingSuccessful`, we concluded that our problem was a classification problem.  
For our classification task, we decided to experiment with three models to determine which performed best on our dataset. The models we chose included **Logistic Regression**, **Random Forest**, and **Gradient Boosting**.  

We decided to split the dataset into three sets:  
- **Training Set**: Useful for the model to learn and adjust parameters; essential for generalization.  
- **Validation Set**: Helpful for tuning hyperparameters and preventing overfitting or underfitting during training.  
- **Test Set**: Reserved for final evaluation to provide an unbiased estimate of the model's performance on unseen data.

The models we chose offered a range of approaches, from linear models to ensemble methods, which were well-suited for a variety of data characteristics. This is why we decided to use and compare them.  
In particular, we decided to use each model for specific characteristics useful for us.

### **How They Work**

- **Logistic Regression**:  
Logistic Regression is a machine learning algorithm used for binary classification problems. It predicts the probability of an input belonging to a particular class (e.g., 0 or 1). It works in this way:
  - **Input Features**: The model combines the input features into a linear equation.
  - **Sigmoid Activation**: The linear result is passed through the Sigmoid function to map it to a probability between 0 and 1.
  - **Prediction**: If the probability `y ≥ 0.5`, classify as `1`. If `y < 0.5`, classify as `0`.

- **Random Forest**:  
Random Forest is an ensemble machine learning algorithm that improves prediction accuracy and robustness by combining multiple decision trees. It works in this way:
  - **Building Trees**: Randomly samples data with replacement (bootstrapping).
  - **Selects Features**: Selects a random subset of features at each split to create diverse trees.
  - **Prediction**: For classification, a majority vote determines the predicted class. For probabilities, the algorithm averages the probabilities from all trees.

- **Gradient Boosting**:  
Gradient Boosting is a machine learning algorithm very used for classification. It builds an ensemble of decision trees sequentially, where each new tree corrects the errors of the previous trees. It works in this way:
  - **Initialize**: Start with an initial prediction (e.g. log-odds).
  - **Build Trees**: Add decision trees one by one. Each tree is trained to correct the residual errors of the previous model.
  - **Combine Models**: Combine the predictions from all trees, weighted by the learning rate, to make the final prediction.

### **Model Evaluation**

We evaluated the models using several metrics:
- **Accuracy**: The proportion of correctly predicted instances. (This is our most used metric)
- **F1-Score**: The harmonic mean of precision and recall.
- **ROC AUC**: Measures the model's ability to distinguish between classes.

### **Comparison of Results**

We optimized and evaluated three classification models using "GridSearchCV" (for Logistic Regression and Random Forest) and "RandomizedSearchCV" (for Gradient Boosting) for hyperparameter tuning. These methods helped identify the best parameters for each model, ensuring better generalization. After tuning, the models were retrained on the full training set and evaluated on a separate test set.

**Model Evaluation**:  
The results indicated that all three models performed exceptionally well, with high accuracy, F1-Score, and ROC-AUC values.  
- **Logistic Regression**: Accuracy = 97.27%  
- **Random Forest**: Accuracy = 96.92%  
- **Gradient Boosting**: Accuracy = 98.03%  

Gradient Boosting stood out as the best model due to its slightly higher accuracy (98.03%), F1-Score (98.03%), and an outstanding ROC-AUC of 99.83%, showcasing its superior ability to capture complex patterns in the data while maintaining excellent generalization. The ROC curve further highlights its robustness, with a near-perfect AUC, making it a reliable choice for this classification task.

   ![Comparing the results through ROC-AUC](roc-auc.png)

---

## **Conclusions**

The analysis conducted in this project demonstrates that aerogel bonding is a predictable and optimizable process, thanks to the application of machine learning techniques. The Gradient Boosting model, in particular, showcased high performance, indicating its potential as a reliable tool for assessing bonding quality.

![top 20 Feature importance](featureimportance.png)

As we can see from the "Top 20 Feature Importances" image, two critical predictors for bonding success were identified: **`BondingRiskRating`**, a key indicator of process risk, and **`ProcessedKilograms`**, which reflects the material handling quality. Importantly, during our long analysis we observed that **`BondingRiskRating`** is inversely proportional to **`BondingSuccessful`**, meaning that lower risk ratings are strongly associated with higher bonding success. 
In addition to these predictors, other features such as **`ByproductRation`**, **`MistakesLastYear`**, and **`RequestedProcessAmount`** also emerged as significant contributors to bonding success, as highlighted in the feature importance analysis. These features emphasize the multifaceted nature of the bonding process, where both material characteristics and operational factors play crucial roles. The feature importance analysis underscored the dominance of **`BondingRiskRating`**, reinforcing its relevance in risk assessment and decision-making processes.
Despite these insights, the dataset presented two major challenges: 
1. **Class Imbalance**: The target variable **`BondingSuccessful`** was heavily skewed. This imbalance necessitated careful handling. 
2. **Presence of Outliers**: Variables such as **`ProcessedKilograms`** and **`RequestedProcessAmount`** exhibited significant variability and outliers, reflecting operational inconsistencies that may require further investigation and standardization to improve process stability.

From a commercialization perspective, aerogel bonding shows strong potential for industries such as aerospace and solid-state batteries. To fully leverage this potential, it is essential to optimize processes based on the identified critical factors. For instance, targeted worker training and refined material handling protocols can address variability in operational conditions. Additionally, mitigating risks through consistent quality control and operational standardization will ensure that the bonding process remains reliable across diverse applications. Extending the analysis to larger and more diverse datasets will also enhance the generalizability and robustness of the findings, facilitating their adoption in real-world industrial scenarios.
In summary, this project has laid a solid foundation for understanding and optimizing aerogel bonding. The insights gained, coupled with the proposed recommendations, position the process for effective commercialization. With further refinement and the implementation of the suggested measures, aerogel bonding can meet industrial demands with high reliability and performance, addressing critical challenges in advanced manufacturing and thermal management applications.




Thank you for your attention !!!

Francesco Costa, Lorenzo Cinelli, Sebastien Jr Foumane 
