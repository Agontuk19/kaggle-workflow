# Titanic Survival Prediction
Survival prediction on the **Titanic dataset** using **Gradient Boosting**.

### Workflow
1. **Feature Engineering**
   - **Title:** Extracted from the passenger Name and categorized into Mr, Mrs, Miss, Master, and Rare.
   - **Family Size:** Calculated as the sum of SibSp and Parch.
   - **Is Alone:** A binary indicator identifying whether a passenger was traveling alone.
   - **Deck:** Extracted as the first character of Cabin, with missing values assigned to Missing.
   - **Ticket Group:** Calculated as the number of passengers sharing the same ticket number.
   - **Age Imputation:** Missing Age values were imputed using a Random Forest Regressor trained on relevant features, including Title, Pclass, SibSp, Parch, and Fare, instead of using simple mean or median imputation.

2. **Preprocessing**
   - A **ColumnTransformer** was used to apply separate preprocessing pipelines based on feature type.
   - The one-hot encoding process used **handle_unknown="ignore"** to ensure that previously unseen categories would not cause errors during prediction.

3. **Model Selection**
   - Compared LogisticRegression, RandomForest, GradientBoosting, XGBoost, LightGBM via 5-fold **StratifiedKFold** CV
   - **GradientBoosting** performed best

4. **Hyperparameter Tuning**
   - **RandomizedSearchCV** (40 iterations, 5-fold CV) on GradientBoosting

5. **Evaluation**
   - Final model selected by CV mean accuracy (not a single train/val split, which is noisy on this small dataset)

### Result
**Accuracy: 85%**

### Tools
Python • NumPy • Pandas • Scikit-learn
