import pandas as pd
import numpy as np 
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load data
df = pd.read_csv('train.csv')

# 2. Covert 'bdate' to age
def calculate_age(bdate):
    try:
        year = int(str(bdate).split('.')[-1])
        return datetime.now().year - year if year > 1900 else np.nan
    except:
        return np.nan

df['age'] = df['bdate'].apply(calculate_age)
df['age'].fillna(df['age'].median(), inplace = True)

# 3. Drop unused columns
df.drop(columns=['bdate', 'langs', 'last_seen', 'occupation_name', 'd'], inplace = True, errors = 'ignore')

# 4. Fill missing values
num_cols = df.select_dtypes(include=['float64', 'int64']).columns
cat_cols = df.select_dtypes(include='object').columns

for col in num_cols:
    df[col].fillna(df[col].median(), inplace = True)
for col in cat_cols:
    df[col].fillna(df[col].mode()[0], inplace = True)

# 5. Convert boolean strings to integers
for col in ['has_photo', 'has_mobile', 'life_main', 'people_main']:
    df[col] = df[col].astype(str).str.lower().replace({'true': 1, 'false': 0})
    df[col] = df[col].astype(float).fillna(0).astype(int)

# 6. Encode categorial columns
label_cols = ['sex', 'education_form', 'education_status', 'city', 'occupation_type']
for col in label_cols:
    df[col] = df[col].astype(str)
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# 7. Select features and target
features  = [
    'sex', 'age', 'has_photo', 'has_mobile', 'followers_count', 'graduation',
    'education_form', 'relation', 'education_status', 'life_main',
    'people_main', 'city', 'occupation_type', 'career_start', 'career_end'
]

X = df[features].copy()

#konversi string ke angka secara aman
for col in X.columns:
    X[col] = X[col].astype(str).str.lower().replace({'true': 1, 'false': 0, 'nan': 0, 'unknown': 0})
    X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0).astype(float)
y = df['result']

# 8. Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 9. Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 10. Train KNN model
knn = KNeighborsClassifier(n_neighbors=5, metric = 'manhattan')
knn.fit(X_train, y_train)

# 11. Predict and evaluate
y_pred = knn.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"Accuracy: {acc*100:.2f}%")
print("Classfication Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))