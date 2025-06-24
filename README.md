# Disease-Diagnosis-PredictionDiabetes Prediction Model - Code & Dataset Documentation
📁 Dataset Information
Source: PIMA Indians Diabetes Database from Kaggle
Download Command: kaggle datasets download -d uciml/pima-indians-diabetes-database
File: diabetes.csv

Dataset Structure
text
pregnancies,glucose,blood_pressure,skin_thickness,insulin,bmi,diabetes_pedigree,age,outcome
6,148,72,35,0,33.6,0.627,50,1
1,85,66,29,0,26.6,0.351,31,0
...
Key Characteristics
768 instances (500 negative, 268 positive)

8 medical features + 1 binary outcome

Missing values: Represented as zeros in medical fields

Class imbalance: 65% negative vs 35% positive cases

🖥️ Code Architecture
1. Data Processing Pipeline (data_preprocessing.py)
python
def preprocess_data(df):
    # Handle missing values (zeros)
    medical_features = ['glucose', 'blood_pressure', ...]
    df[medical_features] = df[medical_features].replace(0, np.nan)
    
    # Impute with median
    for col in medical_features:
        df[col].fillna(df[col].median(), inplace=True)
    
    # Feature selection
    selector = SelectKBest(f_classif, k=5)
    X_selected = selector.fit_transform(X, y)
    
    return X_selected, y
2. Model Training (train.py)
python
def train_model(X, y):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(...)
    
    # Handle imbalance
    smote = SMOTE()
    X_res, y_res = smote.fit_resample(X_train, y_train)
    
    # Initialize models
    models = {
        'gb': GradientBoostingClassifier(),
        'svm': SVC(probability=True),
        'mlp': MLPClassifier()
    }
    
    # Train and evaluate
    for name, model in models.items():
        model.fit(X_res, y_res)
        evaluate(model, X_test, y_test)
    
    return best_model
3. Evaluation Metrics (evaluation.py)
python
def evaluate(model, X_test, y_test):
    # Generate predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]
    
    # Calculate metrics
    print(classification_report(y_test, y_pred))
    print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.2f}")
    
    # Plot confusion matrix
    plot_confusion_matrix(...)
    
    # SHAP analysis (for interpretability)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test)
🔄 Workflow
Data Loading

python
df = pd.read_csv('data/diabetes.csv')
Preprocessing

python
X, y = preprocess_data(df)
Model Training

python
model = train_model(X, y)
Evaluation

python
evaluate(model, X_test, y_test)
Saving Model

python
joblib.dump(model, 'models/gb_model.pkl')
⚠️ Important Notes
Data Requirements:

Kaggle API key needed for initial download

Minimum 1GB RAM recommended for SHAP analysis

Runtime Considerations:

Full training takes ~5 minutes on CPU

SHAP analysis adds ~2 minutes per model

Reproducibility:

python
np.random.seed(42)
random_state = 42  # Used in all sklearn functions
Dependencies:

Primary: scikit-learn, pandas, numpy

Optional: shap (for explainability), imbalanced-learn

