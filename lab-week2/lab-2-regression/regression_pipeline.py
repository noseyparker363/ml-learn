"""
End-to-End Machine Learning Regression Pipeline
===============================================
วิชา: CIT0013 Machine Learning | Lab 2: Penguin Body Mass Prediction (Regression)
งานนี้: ทำนายน้ำหนักตัวของนกเพนกวิน ( body_mass_g ) ซึ่งเป็นข้อมูลแบบต่อเนื่อง (Continuous Variable)
Dataset: Penguins จาก seaborn (344 rows, 7 columns)

ความแตกต่างสำคัญกับ Lab 1 (Titanic Classification):
1. Target (y) เป็นค่าต่อเนื่องเชิงตัวเลข ไม่ใช่สวิตช์เปิด-ปิด (0 หรือ 1)
2. ขั้นตอนแบ่งกลุ่มข้อมูล (Train/Test Split) ในงาน Regression จะไม่มีการใช้พารามิเตอร์ `stratify`
3. เกณฑ์ชี้วัดประสิทธิภาพจะใช้ตัวชี้วัดของ Regression ได้แก่ MAE, RMSE และ R-squared (R2)
4. โมเดลที่จูนและนำมาเปรียบเทียบจะเป็นฝั่ง Regressor
"""

# ============================================================
# STEP 1: นำเข้า Libraries
# ============================================================
import seaborn as sns
import pandas as pd
import numpy as np
import logging
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import warnings
warnings.filterwarnings("ignore")

# ตั้งค่า Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("penguins-ml")

for noisy in ("matplotlib", "seaborn"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

log.info("STEP 1: นำเข้า Libraries สำหรับโมเดล Regression เสร็จเรียบร้อย")


# ============================================================
# STEP 2: โหลดข้อมูล Penguins จาก Seaborn
# ============================================================
penguins = sns.load_dataset("penguins")

log.info(f"STEP 2: โหลดข้อมูลสำเร็จ ขนาดข้อมูล={penguins.shape}")
log.info(f"รายชื่อคุณลักษณะ (Columns) ทั้งหมด: {list(penguins.columns)}")
print("\n--- ตัวอย่างข้อมูลนกเพนกวิน 5 แถวแรก ---")
print(penguins.head().to_string())


# ============================================================
# STEP 3: ตรวจสอบข้อมูลเบื้องต้น (Quick Look)
# ============================================================
log.info("=" * 60)
log.info("STEP 3: สำรวจโครงสร้างข้อมูลเบื้องต้นและค่าสูญหาย")
log.info("=" * 60)

log.info(f"ค่าน้ำหนักนกเพนกวินเฉลี่ย (Target Mean): {penguins['body_mass_g'].mean():.2f} กรัม")
log.info(f"น้ำหนักต่ำสุด: {penguins['body_mass_g'].min()} กรัม | น้ำหนักสูงสุด: {penguins['body_mass_g'].max()} กรัม")

missing_data = penguins.isnull().sum()
print("\n--- สรุปจำนวนข้อมูลสูญหาย (Missing Values) รายคอลัมน์ ---")
print(missing_data[missing_data > 0].to_string())


# ============================================================
# STEP 4: เลือก Features และ Target แล้วแบ่ง Train/Test Set
# ============================================================
log.info("=" * 60)
log.info("STEP 4: เลือก Features และแบ่งกลุ่มข้อมูลสำหรับ Train / Test")
log.info("=" * 60)

# Target คือ 'body_mass_g' (น้ำหนักตัวเป็นกรัม - ข้อมูลต่อเนื่อง)
# Features ที่ใช้ทำนาย: species (สายพันธุ์), island (เกาะที่อยู่อาศัย), 
# bill_length_mm (ความยาวจะงอยปาก), bill_depth_mm (ความหนาจะงอยปาก), 
# flipper_length_mm (ความยาวครีบกระพือปีก), sex (เพศ)
feature_cols = ["species", "island", "bill_length_mm", "bill_depth_mm", "flipper_length_mm", "sex"]

# ตัดแถวที่ Target เป็นค่าว่าง (NaN) ออกก่อนเพื่อไม่ให้เกิดข้อผิดพลาดในการฝึกฝนโมเดล
penguins_clean = penguins.dropna(subset=["body_mass_g"]).copy()

X = penguins_clean[feature_cols]
y = penguins_clean["body_mass_g"]

log.info(f"ขนาดข้อมูลหลังลบแถวที่ target ว่าง: {X.shape[0]} แถว (จากเดิม {penguins.shape[0]})")

# แบ่งข้อมูลออกเป็นชุดฝึกสอน (Train Set) 80% และชุดทดสอบ (Test Set) 20%
# *** ข้อสังเกตหลักสำหรับนักศึกษา: งาน Regression จะไม่มีการใส่ stratify=y เหมือน Classification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

log.info(f"ชุดฝึกสอน (Train Set): {X_train.shape[0]} แถว | ชุดทดสอบ (Test Set): {X_test.shape[0]} แถว")


# ============================================================
# STEP 5: สร้างกระบวนการเตรียมข้อมูลแบบท่อส่ง (Preprocessing Pipeline)
# ============================================================
log.info("=" * 60)
log.info("STEP 5: สร้าง Preprocessing Pipeline")
log.info("=" * 60)

numeric_features = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm"]
categorical_features = ["species", "island", "sex"]

# ท่อส่งสำหรับตัวเลข: เติมค่าว่างด้วย Median และปรับสเกลมาตรฐานด้วย StandardScaler
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# ท่อส่งสำหรับข้อมูลหมวดหมู่: เติมค่าว่างด้วย Mode และทำ One-Hot Encoding
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

# รวมท่อส่งย่อยเข้าด้วยกัน
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

log.info("สร้าง Preprocessor Pipeline สำหรับงาน Regression เรียบร้อย")


# ============================================================
# STEP 6: เปรียบเทียบผลลัพธ์ของ 3 สถาปัตยกรรมโมเดล ด้วย Cross-Validation
# ============================================================
log.info("=" * 60)
log.info("STEP 6: เปรียบเทียบโมเดลทางเลือก (5-fold Cross-Validation)")
log.info("=" * 60)

# โมเดลฝั่ง Regressor สำหรับทำนายค่าตัวเลขต่อเนื่อง
models = {
    "Linear Regression":       LinearRegression(),
    "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42)
}

results = {}
for name, model in models.items():
    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    # เราจะใช้เกณฑ์ชี้วัดเป็นความผิดพลาดสัมบูรณ์เฉลี่ยแบบติดลบ (neg_mean_absolute_error) ในการเปรียบเทียบในขั้นตอน CV
    scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="neg_mean_absolute_error")
    # แปลงคะแนนกลับเป็นค่าบวก (MAE ยิ่งน้อยยิ่งดี)
    mae_scores = -scores
    results[name] = mae_scores
    log.info(f"  {name:25s}: MAE เฉลี่ย = {mae_scores.mean():.2f} กรัม (+/- ค่าแกว่ง {mae_scores.std():.2f})")

# ค้นหาโมเดลที่ให้ค่าความคลาดเคลื่อน MAE ต่ำสุด (ดีที่สุด)
best_model_name = min(results, key=lambda k: results[k].mean())
best_mae = results[best_model_name].mean()
log.info(f"⭐ โมเดลที่ดีที่สุดจากการสแกน CV: {best_model_name} (MAE ต่ำที่สุดเฉลี่ย = {best_mae:.2f} กรัม)")


# ============================================================
# STEP 7: ค้นหาพารามิเตอร์ของโมเดลตัวที่ดีที่สุดโดยอัตโนมัติ (GridSearchCV)
# ============================================================
log.info("=" * 60)
log.info("STEP 7: ค้นหาพารามิเตอร์ที่ดีที่สุด (Hyperparameter Tuning)")
log.info("=" * 60)

# กำหนดตารางจูนพารามิเตอร์สำหรับโมเดลแต่ละกลุ่ม
param_grids = {
    "Linear Regression": {
        "model__fit_intercept": [True, False]
    },
    "Decision Tree Regressor": {
        "model__max_depth":         [None, 5, 10],
        "model__min_samples_split": [2, 5, 10]
    },
    "Random Forest Regressor": {
        "model__n_estimators":      [50, 100, 150],
        "model__max_depth":         [None, 5, 10],
        "model__min_samples_split": [2, 5]
    }
}

best_model_class = type(models[best_model_name])
selected_grid = param_grids.get(best_model_name)

if selected_grid is None:
    log.warning(f"ไม่พบตารางพารามิเตอร์สำหรับโมเดล {best_model_name} — จะใช้ตารางของ Linear Regression")
    selected_grid = param_grids["Linear Regression"]
    best_model_class = LinearRegression

log.info(f"โมเดลที่จะทำการปรับจูน: {best_model_name}")

final_pipe = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", best_model_class())
])

# จูนโมเดลด้วย GridSearchCV โดยเน้นลดค่า MAE
grid_search = GridSearchCV(
    final_pipe, selected_grid, cv=5,
    scoring="neg_mean_absolute_error", n_jobs=-1
)
grid_search.fit(X_train, y_train)

log.info(f"ค่าพารามิเตอร์ที่เหมาะสมที่สุด (Best params): {grid_search.best_params_}")
log.info(f"ค่าความคลาดเคลื่อน MAE ต่ำสุดที่ได้จากการจูน (Best CV MAE): {-grid_search.best_score_:.2f} กรัม")


# ============================================================
# STEP 8: ประเมินผลประสิทธิภาพขั้นสุดท้ายบนชุดทดสอบ (Test Set)
# ============================================================
log.info("=" * 60)
log.info("STEP 8: ประเมินประสิทธิภาพขั้นสุดท้ายด้วยชุดข้อมูลทดสอบ")
log.info("=" * 60)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# คำนวณค่าสถิติวัดผลของงาน Regression
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n--- รายงานผลวิเคราะห์สถิติโมเดล Regression ---")
print(f"  ความคลาดเคลื่อนสัมบูรณ์เฉลี่ย (MAE):   {mae:.2f} กรัม (โมเดลพยากรณ์คลาดเคลื่อนเฉลี่ย +/- {mae:.2f} กรัม)")
print(f"  รากกำลังสองเฉลี่ยความคลาดเคลื่อน (RMSE): {rmse:.2f} กรัม")
print(f"  ค่าสัมประสิทธิ์การตัดสินใจ (R-squared หรือ R2): {r2:.3f} หรือ {r2:.1%}")
log.info("หมายเหตุประกอบคำอธิบาย: ค่า R-squared บ่งบอกว่า ฟีเจอร์ที่เรานำมาใช้สามารถอธิบายความผันแปรของน้ำหนักตัวนกเพนกวินได้กี่เปอร์เซ็นต์")


# ============================================================
# STEP 9: วิเคราะห์สัมประสิทธิ์หรือความสำคัญของฟีเจอร์ (Feature Importance / Coefficients)
# ============================================================
log.info("=" * 60)
log.info("STEP 9: ถอดรหัสอิทธิพลของแต่ละปัจจัยต่อโมเดลทำนาย")
log.info("=" * 60)

model_step = best_model["model"]
onehot_step = best_model["preprocessor"].named_transformers_["cat"]["onehot"]
feature_names = numeric_features + list(onehot_step.get_feature_names_out(categorical_features))

if hasattr(model_step, "feature_importances_"):
    importances = model_step.feature_importances_
    log.info("ดึงผลด้วยคุณสมบัติ: .feature_importances_ (Tree-based Model)")
elif hasattr(model_step, "coef_"):
    importances = model_step.coef_
    log.info("ดึงผลด้วยคุณสมบัติ: .coef_ (Linear Regression Coefficients)")
else:
    importances = None

if importances is not None:
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "value": importances
    })
    
    # หากเป็นสัมประสิทธิ์เชิงเส้น จะมีค่าบวกและลบ (บอกทิศทาง) เราจะจัดเรียงด้วยขนาดความสำคัญ (Absolute value)
    if hasattr(model_step, "coef_"):
        importance_df["abs_value"] = importance_df["value"].abs()
        importance_df = importance_df.sort_values("abs_value", ascending=False).drop(columns=["abs_value"])
    else:
        importance_df = importance_df.sort_values("value", ascending=False)

    print("\n--- ตารางอิทธิพลของฟีเจอร์ต่อการทำนายค่าน้ำหนัก ---")
    print(importance_df.to_string(index=False))


# ============================================================
# STEP 9.5: บันทึกโมเดลและระบบตรวจสอบข้อมูลขาเข้า (Validation & Serialization)
# ============================================================
log.info("=" * 60)
log.info("STEP 9.5: บันทึกโมเดลและจัดเตรียมระบบความปลอดภัยข้อมูลขาเข้า")
log.info("=" * 60)

MODEL_PATH = Path("penguins_best_model.joblib")

def save_model(model, path: Path = MODEL_PATH) -> None:
    joblib.dump(model, path)
    log.info(f"บันทึกไฟล์โมเดลไว้เรียบร้อยที่: {path.resolve()}")

# โครงสร้างตัวแปรของสายพันธุ์ เกาะ และเพศที่ถูกต้อง
VALID_SPECIES = {"Adelie", "Chinstrap", "Gentoo"}
VALID_ISLANDS = {"Torgersen", "Biscoe", "Dream"}
VALID_SEX = {"male", "female"}
REQUIRED_COLS = ["species", "island", "bill_length_mm", "bill_depth_mm", "flipper_length_mm", "sex"]

def validate_penguin_input(df: pd.DataFrame) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("ข้อมูลขาเข้าต้องส่งในรูปแบบ DataFrame")
        
    missing_cols = set(REQUIRED_COLS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"ข้อมูลขาดคอลัมน์สำคัญดังนี้: {sorted(missing_cols)}")
        
    # ตรวจสอบขอบเขตข้อมูลหมวดหมู่
    bad_species = set(df["species"].dropna().unique()) - VALID_SPECIES
    if bad_species:
        raise ValueError(f"สายพันธุ์นกเพนกวินไม่ถูกต้อง: {bad_species}")
        
    bad_island = set(df["island"].dropna().unique()) - VALID_ISLANDS
    if bad_island:
        raise ValueError(f"ชื่อเกาะไม่ถูกต้อง: {bad_island}")

    bad_sex = set(df["sex"].dropna().unique()) - VALID_SEX
    if bad_sex:
        raise ValueError(f"เพศไม่ถูกต้อง: {bad_sex}")

# บันทึกโมเดล
save_model(best_model)


# ============================================================
# STEP 10: จำลองทำนายน้ำหนักตัวของนกเพนกวินตัวใหม่
# ============================================================
log.info("=" * 60)
log.info("STEP 10: จำลองทำนายน้ำหนักตัวนกเพนกวินตัวใหม่")
log.info("=" * 60)

# จำลองนกเพนกวินตัวใหม่
new_penguins = pd.DataFrame({
    "species":           ["Adelie", "Gentoo"],
    "island":            ["Torgersen", "Biscoe"],
    "bill_length_mm":    [39.1, 48.2],
    "bill_depth_mm":     [18.7, 14.3],
    "flipper_length_mm": [181.0, 210.0],
    "sex":               ["male", "female"]
})

validate_penguin_input(new_penguins)

predicted_mass = best_model.predict(new_penguins)

print("\n--- ข้อมูลนกเพนกวินตัวใหม่ที่นำมาทำนาย ---")
print(new_penguins.to_string(index=False))

print("\n--- ผลลัพธ์วิเคราะห์น้ำหนักโดยประมาณ ---")
for i, pred in enumerate(predicted_mass, start=1):
    log.info(f"นกเพนกวินตัวที่ {i}: ทำนายน้ำหนักได้ = {pred:.2f} กรัม (ประมาณ {pred/1000:.2f} กิโลกรัม)")

log.info("=" * 60)
log.info("เสร็จสมบูรณ์! กระบวนการสร้างและประเมินผล ML Regression Pipeline ดำเนินการเสร็จสมบูรณ์")
log.info("=" * 60)
