"""
End-to-End Machine Learning Pipeline
====================================
งานนี้: ทำนายว่าผู้โดยสาร Titanic รอดชีวิตหรือไม่ (Binary Classification)
Dataset: Titanic จาก seaborn (891 rows, 15 columns)
ผู้เรียนสามารถรันทีละ Step เพื่อทำความเข้าใจทีละขั้น
"""

# ============================================================
# STEP 1: นำเข้า Libraries ที่ต้องใช้
# ============================================================
import seaborn as sns
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import warnings
warnings.filterwarnings("ignore")

print("=" * 60)
print("STEP 1: นำเข้า Libraries เสร็จเรียบร้อย")
print("=" * 60)


# ============================================================
# STEP 2: โหลดข้อมูล Titanic
# ============================================================
titanic = sns.load_dataset('titanic')

print(f"\nDataset shape: {titanic.shape}")
print(f"Columns: {list(titanic.columns)}")
print("\nตัวอย่าง 5 แถวแรก:")
print(titanic[['survived', 'pclass', 'sex', 'age', 'fare', 'embarked']].head())


# ============================================================
# STEP 3: ตรวจสอบข้อมูลเบื้องต้น (Quick Look)
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Quick Look at Data")
print("=" * 60)

print(f"\nอัตราการรอดชีวิตโดยรวม: {titanic['survived'].mean():.1%}")
print(f"\nจำนวน Missing values ต่อ column:")
print(titanic.isnull().sum())


# ============================================================
# STEP 4: เลือก Features และ Target แล้วแบ่ง Train/Test
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: เลือก Features และแบ่ง Train/Test Set")
print("=" * 60)

# Target คือ 'survived' (0 = เสียชีวิต, 1 = รอดชีวิต)
# Features ที่ใช้: pclass, sex, age, fare, embarked
feature_cols = ['pclass', 'sex', 'age', 'fare', 'embarked']
X = titanic[feature_cols].copy()
y = titanic['survived']

print(f"\nFeatures ที่ใช้: {feature_cols}")
print(f"Target: 'survived'")
print(f"X shape: {X.shape}, y shape: {y.shape}")

# แบ่งข้อมูล 80% train, 20% test โดยใช้ stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain set: {X_train.shape[0]} rows")
print(f"Test set:  {X_test.shape[0]} rows")
print(f"Train survival rate: {y_train.mean():.1%}")
print(f"Test survival rate:  {y_test.mean():.1%}")


# ============================================================
# STEP 5: สร้าง Preprocessing Pipeline
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: สร้าง Preprocessing Pipeline")
print("=" * 60)

# แยก features เป็น numeric กับ categorical
numeric_features = ['age', 'fare']
categorical_features = ['pclass', 'sex', 'embarked']

# Pipeline สำหรับ numeric: เติมค่าว่างด้วย median แล้ว scale
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Pipeline สำหรับ categorical: เติมค่าว่างด้วย mode แล้ว one-hot encode
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# รวม pipeline ทั้งสองเข้าด้วยกัน
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

print("\nNumeric features:", numeric_features)
print("Categorical features:", categorical_features)
print("Preprocessor พร้อมใช้งาน")


# ============================================================
# STEP 6: เปรียบเทียบ 3 โมเดลด้วย Cross-Validation
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: เปรียบเทียบ 3 โมเดล (5-fold Cross-Validation)")
print("=" * 60)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree':       DecisionTreeClassifier(random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42)
}

results = {}
for name, model in models.items():
    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='accuracy')
    results[name] = scores
    print(f"  {name:20s}: {scores.mean():.3f} (+/- {scores.std():.3f})")

# หาโมเดลที่ดีที่สุด
best_model_name = max(results, key=lambda k: results[k].mean())
print(f"\nโมเดลที่ดีที่สุด: {best_model_name}")


# ============================================================
# STEP 7: ปรับ Hyperparameters ด้วย GridSearchCV
# ============================================================
print("\n" + "=" * 60)
print("STEP 7: Fine-tune Hyperparameters (GridSearchCV)")
print("=" * 60)

param_grid = {
    'model__n_estimators': [50, 100, 200],
    'model__max_depth': [None, 5, 10],
    'model__min_samples_split': [2, 5]
}

final_pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier(random_state=42))
])

grid_search = GridSearchCV(
    final_pipe, param_grid, cv=5,
    scoring='accuracy', n_jobs=-1
)
grid_search.fit(X_train, y_train)

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best CV accuracy: {grid_search.best_score_:.3f}")


# ============================================================
# STEP 8: ประเมินผลบน Test Set (ครั้งเดียวเท่านั้น!)
# ============================================================
print("\n" + "=" * 60)
print("STEP 8: ประเมินผลบน Test Set")
print("=" * 60)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

test_accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {test_accuracy:.3f}")
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"  TN={cm[0,0]:3d}  FP={cm[0,1]:3d}")
print(f"  FN={cm[1,0]:3d}  TP={cm[1,1]:3d}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['เสียชีวิต', 'รอดชีวิต']))


# ============================================================
# STEP 9: ดู Feature Importance
# ============================================================
print("\n" + "=" * 60)
print("STEP 9: Feature Importance")
print("=" * 60)

importances = best_model['model'].feature_importances_
feature_names = (numeric_features +
                 list(best_model['preprocessor']
                      .named_transformers_['cat']['onehot']
                      .get_feature_names_out(categorical_features)))

importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)

print("\nTop features:")
print(importance_df.to_string(index=False))


# ============================================================
# STEP 10: ทำนายผู้โดยสารใหม่ (ตัวอย่าง)
# ============================================================
print("\n" + "=" * 60)
print("STEP 10: ทำนายผู้โดยสารใหม่")
print("=" * 60)

# สมมติผู้โดยสารใหม่
new_passengers = pd.DataFrame({
    'pclass':  [1, 3, 2],
    'sex':     ['female', 'male', 'female'],
    'age':     [25, 30, 40],
    'fare':    [80.0, 7.5, 20.0],
    'embarked': ['C', 'S', 'Q']
})

predictions = best_model.predict(new_passengers)
probabilities = best_model.predict_proba(new_passengers)[:, 1]

print("\nข้อมูลผู้โดยสาร:")
print(new_passengers.to_string(index=False))
print("\nผลการทำนาย:")
for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
    result = "รอดชีวิต" if pred == 1 else "เสียชีวิต"
    print(f"  ผู้โดยสาร {i+1}: {result} (ความน่าจะเป็น: {prob:.1%})")

print("\n" + "=" * 60)
print("เสร็จสมบูรณ์!  ML Pipeline ทำงานครบทุก Step")
print("=" * 60)
