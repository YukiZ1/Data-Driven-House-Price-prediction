### The code contains four parts: 
  1. Data Cleaning and Variable Selection
  2. Model Selection: Regularized Rgression Model or XGBoost model
  3. Model interpretation
  4. prediction on testing dataset

### Data Cleaning
The training datasets we used has 279 samples. They were scraped from public real estate databases for three target neighborhoods, containing missing fields due to source variations.
(1) there are some unavailable fields, so we will not take into account these variables: 
Street, LandContour, grade to building, LandSlope, Neighborhood, YearRemodAdd, BsmtExposure, BsmtUnfSF, KitchenAbvGr.

(2) Then split concatenated variables
Split Exterior into three variables: Exterior1st, ExterQual, ExterCond;
Split LotInfo into these four variables: LotConfig, LotShape, LotFrontage, LotArea

(3) Next, handle Missing and Empty Values, and convert some variables to numerical form, and create some new variables.
based on the data description document,
- **GarageType**: the Missing values indicate no garage → filled with `"noGarage"` instead of mode.
- **LotFrontage**: Converted to numerical.
- **LotArea**: Converted to numerical; missing values imputed with the mean.
- **BsmtCond**: Missing values indicate no basement → filled with `"noBasement"` for easier encoding.
- **BsmtQual**: Missing values filled with the most frequent value (mode).
- **BsmtFinType1**: Missing values filled with the most frequent value (mode)
- then create a new variable, house age=YrSold-YearBuilt, and there is a negative value, which is wrong record. here we will take absolute value, and keep the row.
- also create new variables neighborhood and sample_id for the data.

(4) Then encoding the categorical predictors to numerical format. This is an important step to ensure the prediction model can work well.

(5) Exploratory Data Analysis (EDA)
The count, mean, median, standard deviation, minimum, maximum, Q1, and Q3 of numerical variables. we can see that the average house price is 158035.
Frequency distributions of categorical variables prior to encoding.

### Variable selection
After data cleaning, we left 38 predictors.
and then based on the multi linear analysis, the variable Utilities_ is perfectly linearly dependent, causing multicollinearity in the model. so we delete it.
and there are other 5 high linearly dependent variables, we will try models with and without these variables.

### Regularized Rgression Model training
In this project, we build 8 different models, using 4 algorithms: the ridge regressionn, Lasso regression, elastic regression, and XGBoost. 
and we have 2 sets of predictors, one include all the variables, the other just include linearly independent variables.
all algorithms will apply to these two sets of predictors.

And we use 10-fold cross validation to evaluate the model performance based on the root mean square error. Results showed that the best model is the boosted tree model with all predictors, achieving lowest root mean square error(RMSE) 24604.36, PRMSE 15.57%, R^2 0.87;

the tunning process follows rules: 
  learning rate (eta): 0.05, 0,10, 0.30
  Max_depth: 3, 5, 7
  Reiterate round (nrounds): 100, 200
  Fixed parameters: Gamma=0, colsample_bytree=0.8, min_child_weight=1, subsample=0.8

final model uses the parameters:
  nrounds         = 100
  max_depth       = 5
  eta             = 0.05
  gamma           = 0
  colsample_bytree= 0.8
  min_child_weight= 1
  subsample       = 0.8

### Model Interpretation
The variable importance plot showed that the top three important variables are overall quality, above ground living area, house age.
<img width="1118" height="854" alt="image" src="https://github.com/user-attachments/assets/10ae7562-bcf2-492a-bdec-91a86acbe452" />

more specifically,
houses with higher quality ratings, generally sell for higher prices.
larger living areas tend to be associated with higher prices.
older houses generally having lower market values.
<img width="1006" height="1108" alt="image" src="https://github.com/user-attachments/assets/eb662919-2f8f-433b-b7cf-a3c79daf0a96" />

### predict
we can use this model to predict the sale price of other houses.
you can see the csv document as final result.[xgb_predictions.csv](https://github.com/user-attachments/files/21807128/xgb_predictions.csv)

### end
Thanks Neal for the trust in our technical team. Please feel free to reach out anytime if you have any questions.
