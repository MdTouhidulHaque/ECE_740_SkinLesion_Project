from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import numpy as np 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pickle

# Load the training and test data
train_data = np.load('../extracted_features/train_extracted_features/train_extracted_features.npz', allow_pickle=True)
X_train = train_data['features']   
y_train = train_data['labels']   

test_data = np.load('../extracted_features/balanced_test_extracted_features/balanced_test_extracted_features.npz', allow_pickle=True)
#test_data = np.load('../extracted_features/independent_test_extracted_features/independent_test_extracted_features.npz', allow_pickle=True)
X_test = test_data['features']   
y_test = test_data['labels']

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Encode the string labels into numeric labels
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Classifiers dictionary
classifiers = {
    'Logistic Regression': LogisticRegression(random_state=9,max_iter=1000),
    'LDA': LinearDiscriminantAnalysis(solver='svd'),
    'KNN': KNeighborsClassifier(n_neighbors=9),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(n_estimators=200, random_state=0),
    'Gaussian NB': GaussianNB(var_smoothing=1e-9),
    'SVM': SVC(kernel='linear', C=1, random_state=0)
}

results = {}
for name, clf in classifiers.items():
    # Fit model
    clf.fit(X_train, y_train_encoded)
    
    # Predict using the trained model
    y_pred_encoded = clf.predict(X_test)

    # Calculate metrics
    results[name] = {
        'accuracy': accuracy_score(y_test_encoded, y_pred_encoded),
        'precision': precision_score(y_test_encoded, y_pred_encoded, average='weighted'),
        'recall': recall_score(y_test_encoded, y_pred_encoded, average='weighted'),
        'f1_score': f1_score(y_test_encoded, y_pred_encoded, average='weighted')
    }
    cm = confusion_matrix(y_test_encoded, y_pred_encoded)
    print(f"Confusion Matrix for {name}:")
    print(cm)

# Output results
print(results)

# Save results
with open('../results/ml_models_results/classification_results.pkl', 'wb') as file:
    pickle.dump(results, file)
