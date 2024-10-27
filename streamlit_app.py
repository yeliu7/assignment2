import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold, TunedThresholdClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn import metrics

# Load the data
df = pd.read_csv('https://raw.githubusercontent.com/yeliu7/assignment2/refs/heads/main/kickstarter_2016.csv')
df = df[(df['State'] != "Live") & (df['Goal'] > 0)]

# Create new target variable (success)
# axis=1 means to apply the function to each row
def success(row):
    if row['State'] == "Successful":
        return 1
    else:
        return 0
df['Success'] = df.apply(success, axis=1)

# Create new features: log of funding goal, campaign duration, project name length in words
df['Log_Goal'] = np.log10(df['Goal'])
df['Deadline'] = pd.to_datetime(df['Deadline'])
df['Launched'] = pd.to_datetime(df['Launched'])
df['Duration'] = (df['Deadline'] - df['Launched']).dt.days
df['Name_Length'] = df['Name'].apply(lambda x: len(str(x).split()))

# numerical features:
num_features = ['ID', 'Log_Goal', 'Duration', 'Name_Length', 'Pledged', 'Backers']
num_features_and_target = num_features + ['Success']

# Show a correlation matrix of numerical features
corr = df[num_features_and_target].corr()

# Show correlation as a heatmap
import seaborn as sns
fig, ax = plt.subplots(figsize=(6.4, 4.8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
ax.set_title("Correlation Matrix")
st.pyplot(fig)

# Choose useful features to predict campaign success and exclude features that
# are not available at the time of launch
df = df.drop(['Goal','Launched','Deadline','State','Pledged', 'Backers', 'ID', 'Name'], axis=1)

st.write(df)

# Sidebar: Classifier selection
st.sidebar.header("Choose Classifier")
classifier_name = st.sidebar.selectbox(
    'Choose Classifier',
    ('Logistic Regression', 'Random Forest', 'Gradient Boosting')
)

# Sidebar: Feature selection
st.sidebar.header("Feature Selection")
all_features = ['Log_Goal', 'Duration', 'Name_Length', 'Category', 'Subcategory','Country']
selected_features = st.sidebar.multiselect(
    'Select features:',
    all_features,
    default=all_features
)

if not selected_features:
    st.error("Select feature.")
    st.stop()

# Split the dataset into X and y
X = df[selected_features]
y = df['Success']

# Identify features based on user selection
num_features = [f for f in selected_features if f in ['Log_Goal', 'Duration', 'Name_length']]
cat_features = [f for f in selected_features if f in ['Category', 'Subcategory', 'Country']]

# Pipeline for pre-processing numerical features
# Impute missing values with 0
# Scale all numerical features to zero mean and unit variance
num_transformer = Pipeline(steps = [('imputer', SimpleImputer(strategy='constant', fill_value=0)), ('scaler', StandardScaler())])

# Pipeline for pre-processing categorical features
# Impute missing values with 'missing'
# One-hot encode all categorical features
cat_transformer = Pipeline(steps = [('imputer', SimpleImputer(strategy='constant', fill_value='missing')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Pipeline for pre-processing numerical and categorical features
preprocessor = ColumnTransformer(transformers = [('num', num_transformer, num_features), ('cat', cat_transformer, cat_features)])

# three classifiers
classifiers = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(max_samples=0.1, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

# Create the train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Create a pipeline based on user-selected classifer
model = classifiers[classifier_name]
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])

# Stratified k-fold cross-validation
cv = StratifiedKFold(n_splits=5)
scores = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=['accuracy', 'precision', 'recall', 'f1'])

# Show the evaluation metrics
df_scores = pd.DataFrame(scores)
df_scores.loc['mean'] = df_scores.mean()
st.table(df_scores)

# fit and test the model
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

