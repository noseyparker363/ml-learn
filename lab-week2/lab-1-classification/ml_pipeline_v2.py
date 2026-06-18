"""
End-to-End Machine Learning Pipeline (v2 - Production-Ready)
=============================================================
วิชา: CIT0013 Machine Learning | Lab 1: Titanic Survival Prediction
งานนี้: ทำนายว่าผู้โดยสาร Titanic รอดชีวิตหรือไม่ (Binary Classification)
Dataset: Titanic จาก seaborn (891 rows, 15 columns)

เวอร์ชัน v2 นี้ได้รับการปรับปรุงตามแนวทางปฏิบัติที่ดีที่สุด (Best Practices):
1. ย้าย import ทั้งหมดไปไว้ด้านบนสุดเพื่อความสะอาดเรียบร้อย (Clean Code)
2. ปรับปรุงคอมเมนต์คำอธิบายให้เป็นมิตรและเป็นระบบ เหมาะสำหรับนักศึกษาเรียนรู้
3. ปรับ Step 7 ให้ทำโมเดลจูนนิ่ง (Hyperparameter Tuning) แบบไดนามิกตามโมเดลที่ดีที่สุดจาก Step 6
4. ปรับ Step 9 ให้ดึงความสำคัญของฟีเจอร์อย่างปลอดภัย รองรับทั้งโมเดลแบบ Tree-based และ Linear
5. เพิ่มเครื่องมือตรวจสอบความถูกต้องของข้อมูลขาเข้า (Input Validation) และการบันทึกเก็บโมเดล (joblib)
"""

# ============================================================
# STEP 1: นำเข้า Libraries ที่จำเป็นและตั้งค่าระบบ Logging
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
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import warnings
warnings.filterwarnings("ignore")

# ตั้งค่า Logging เพื่อใช้แสดงและบันทึกสถานะการทำงานแทนการใช้ print() ทั่วไป
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("titanic-ml")

# ปิดข้อความเตือนของไลบรารีอื่น ๆ ที่ไม่เกี่ยวกับโมเดลของเรา
for noisy in ("matplotlib", "seaborn"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

log.info("STEP 1: นำเข้า Libraries และตั้งค่าระบบ Logging เสร็จเรียบร้อย")


# ============================================================
# STEP 2: โหลดข้อมูล Titanic จาก Seaborn
# ============================================================
titanic = sns.load_dataset("titanic")

log.info(f"STEP 2: โหลดข้อมูลสำเร็จ ขนาดข้อมูล={titanic.shape}")
log.info(f"รายชื่อคุณลักษณะ (Columns) ทั้งหมด: {list(titanic.columns)}")
print("\n--- ตัวอย่างข้อมูล 5 แถวแรก ---")
print(titanic[["survived", "pclass", "sex", "age", "fare", "embarked"]].head().to_string())


# ============================================================
# STEP 3: ตรวจสอบข้อมูลเบื้องต้น (Quick Look)
# ============================================================
log.info("=" * 60)
log.info("STEP 3: สำรวจโครงสร้างข้อมูลเบื้องต้น")
log.info("=" * 60)

log.info(f"อัตราการรอดชีวิตเฉลี่ยของผู้โดยสารทั้งหมด: {titanic['survived'].mean():.1%}")
missing_data = titanic.isnull().sum()
print("\n--- สรุปจำนวนข้อมูลสูญหาย (Missing Values) รายคอลัมน์ ---")
print(missing_data[missing_data > 0].to_string())


# ============================================================
# STEP 4: เลือกคุณลักษณะ (Features) และเป้าหมาย (Target) แล้วแบ่ง Train/Test Set
# ============================================================
log.info("=" * 60)
log.info("STEP 4: เลือก Features และแบ่งกลุ่มข้อมูลสำหรับ Train / Test")
log.info("=" * 60)

# Target คือ 'survived' (0 = เสียชีวิต, 1 = รอดชีวิต)
# Features ที่นำมาใช้ในการพยากรณ์: pclass, sex, age, fare, embarked
feature_cols = ["pclass", "sex", "age", "fare", "embarked"]
X = titanic[feature_cols].copy()
y = titanic["survived"]

log.info(f"ฟีเจอร์ที่นำมาใช้พยากรณ์: {feature_cols}")
log.info(f"ขนาดของตัวแปรอิสระ X: {X.shape} | ขนาดของตัวแปรตาม y: {y.shape}")

# แบ่งข้อมูลออกเป็นชุดฝึกสอน (Train Set) 80% และชุดทดสอบ (Test Set) 20% โดยใช้ stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

log.info(f"ชุดฝึกสอน (Train Set): {X_train.shape[0]} แถว | ชุดทดสอบ (Test Set): {X_test.shape[0]} แถว")
log.info(f"อัตราการรอดในชุดฝึกสอน: {y_train.mean():.1%} | อัตราการรอดในชุดทดสอบ: {y_test.mean():.1%}")


# ============================================================
# STEP 5: สร้างกระบวนการเตรียมข้อมูลแบบท่อส่ง (Preprocessing Pipeline)
# ============================================================
log.info("=" * 60)
log.info("STEP 5: สร้าง Preprocessing Pipeline สำหรับจัดการข้อมูลสูญหายและแปลงประเภทข้อมูล")
log.info("=" * 60)

# 1) แยกคุณลักษณะออกตามความเหมาะสมของการประมวลผลข้อมูล
# หมายเหตุ: pclass (ชั้นตั๋ว 1, 2, 3) จัดเป็นข้อมูลเชิงอันดับ (Ordinal)
# เราจึงรวบรวมไว้เป็น Numeric เพื่อสเกลแทนการสลับเป็น One-hot เพื่อคงลำดับเอาไว้
numeric_features = ["pclass", "age", "fare"]
categorical_features = ["sex", "embarked"]

# 2) สร้างท่อส่งสำหรับข้อมูลตัวเลข (Numeric Pipeline):
# - เติมข้อมูลที่หายไป (Imputation) ด้วยค่ามัธยฐาน (Median)
# - ปรับสเกลข้อมูลให้เป็นมาตรฐาน (Standardization) ด้วย StandardScaler
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# 3) สร้างท่อส่งสำหรับข้อมูลหมวดหมู่ (Categorical Pipeline):
# - เติมข้อมูลที่หายไปด้วยฐานนิยม (Mode/Most Frequent)
# - แปลงคุณลักษณะเชิงตัวอักษรเป็นเวกเตอร์เลขฐานสอง ด้วย OneHotEncoder
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

# 4) รวมท่อส่งย่อยทั้งสองเข้าด้วยกันตามรายชื่อคอลัมน์ที่กำหนดไว้
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

log.info(f"รายชื่อข้อมูลตัวเลข (Numeric Features): {numeric_features}")
log.info(f"รายชื่อข้อมูลหมวดหมู่ (Categorical Features): {categorical_features}")
log.info("สร้าง Preprocessor Pipeline สำเร็จและพร้อมเริ่มการจัดเรียงชุดข้อมูล")


# ============================================================
# STEP 6: เปรียบเทียบผลลัพธ์ของ 3 สถาปัตยกรรมโมเดล ด้วย Cross-Validation
# ============================================================
log.info("=" * 60)
log.info("STEP 6: เปรียบเทียบโมเดลต่างๆ ด้วย 5-fold Cross-Validation")
log.info("=" * 60)

# กำหนดรายชื่อโมเดลที่ต้องการเปรียบเทียบหาตัวชูโรงหลัก
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree":       DecisionTreeClassifier(random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42)
}

results = {}
for name, model in models.items():
    # สร้างท่อส่งข้อมูลหลักที่รวม Preprocessor และ Model เข้าด้วยกัน
    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    # ประเมินประสิทธิภาพผ่านระบบ Cross Validation 5 ส่วนเท่าๆ กัน
    scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="accuracy")
    results[name] = scores
    log.info(f"  {name:20s}: ค่าความถูกต้องเฉลี่ย = {scores.mean():.3f} (+/- {scores.std():.3f})")

# ค้นหาโมเดลที่มีประสิทธิภาพการเรียนรู้ดีที่สุดแบบไดนามิก
best_model_name = max(results, key=lambda k: results[k].mean())
best_mean = results[best_model_name].mean()
log.info(f"⭐ โมเดลที่ดีที่สุดจากการตรวจสอบเบื้องต้น: {best_model_name} (คะแนน CV เฉลี่ย = {best_mean:.3f})")


# ============================================================
# STEP 7: ค้นหาพารามิเตอร์ของโมเดลตัวที่ดีที่สุดโดยอัตโนมัติ (GridSearchCV)
# ============================================================
log.info("=" * 60)
log.info("STEP 7: ค้นหาพารามิเตอร์ที่ดีที่สุด (Hyperparameter Tuning)")
log.info("=" * 60)

# กำหนดตารางทางเลือกพารามิเตอร์สำหรับโมเดลหลักแต่ละประเภท
param_grids = {
    "Logistic Regression": {
        "model__C":       [0.01, 0.1, 1.0, 10.0],
        "model__penalty": ["l2"],
        "model__solver":  ["lbfgs"],
    },
    "Decision Tree": {
        "model__max_depth":         [None, 5, 10, 15],
        "model__min_samples_split": [2, 5, 10],
        "model__criterion":         ["gini", "entropy"],
    },
    "Random Forest": {
        "model__n_estimators":      [50, 100, 200],
        "model__max_depth":         [None, 5, 10],
        "model__min_samples_split": [2, 5],
    },
}

best_model_class = type(models[best_model_name])
selected_grid = param_grids.get(best_model_name)

# กรณีตรวจไม่พบชุดพารามิเตอร์จูนนิ่งจะดึง Default ของ Random Forest มาสวมสิทธิ์
if selected_grid is None:
    log.warning(f"ไม่พบตารางพารามิเตอร์สำหรับโมเดล {best_model_name} — จะใช้ตารางของ Random Forest เสมือนเป็นโมเดลสำรอง")
    selected_grid = param_grids["Random Forest"]
    best_model_class = RandomForestClassifier

log.info(f"สถาปัตยกรรมโมเดลที่ชนะและนำมาจูน: {best_model_name}")
log.info(f"ตารางช่วงการจูนพารามิเตอร์ (Param Grid): {selected_grid}")

# ดึงโมเดลชนะเข้าร่วมท่อส่งส่งข้อมูลหลัก
final_pipe = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", best_model_class(random_state=42))
])

grid_search = GridSearchCV(
    final_pipe, selected_grid, cv=5,
    scoring="accuracy", n_jobs=-1
)
grid_search.fit(X_train, y_train)

log.info(f"ค่าพารามิเตอร์ที่เหมาะสมที่สุด (Best params): {grid_search.best_params_}")
log.info(f"คะแนนเฉลี่ยจากการจูนสูงสุด (Best CV accuracy): {grid_search.best_score_:.3f}")


# ============================================================
# STEP 8: ประเมินผลประสิทธิภาพขั้นสุดท้ายบนชุดทดสอบ (Test Set)
# ============================================================
log.info("=" * 60)
log.info("STEP 8: ประเมินประสิทธิภาพขั้นสุดท้ายด้วยชุดข้อมูลทดสอบ")
log.info("=" * 60)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

test_accuracy = accuracy_score(y_test, y_pred)
log.info(f"คะแนนความถูกต้องขั้นสุดท้าย (Test Accuracy): {test_accuracy:.3f}")

# คำนวณสรุปตาราง Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\n--- ตารางประเมินผล Confusion Matrix ---")
print(f"  ทำนายเสียชีวิตและผลจริงคือเสียชีวิต (True Negative):   TN = {cm[0,0]:3d}")
print(f"  ทำนายรอดชีวิตแต่ผลจริงคือเสียชีวิต (False Positive):  FP = {cm[0,1]:3d}")
print(f"  ทำนายเสียชีวิตแต่ผลจริงคือรอดชีวิต (False Negative):  FN = {cm[1,0]:3d}")
print(f"  ทำนายรอดชีวิตและผลจริงคือรอดชีวิต (True Positive):   TP = {cm[1,1]:3d}")

# แสดงสถิติด้านความแม่นยำและการจดจำคลาส (Precision, Recall, F1-Score)
print("\n--- รายงานผลวิเคราะห์สถิติโมเดล (Classification Report) ---")
print(classification_report(y_test, y_pred, target_names=["เสียชีวิต", "รอดชีวิต"]))


# ============================================================
# STEP 9: ดึงปัจจัยที่มีความสำคัญสูงสุดต่อการตัดสินใจรอดชีวิต (Feature Importance)
# ============================================================
log.info("=" * 60)
log.info("STEP 9: ถอดรหัสปัจจัยสำคัญสูงสุดต่อโมเดลการประเมิน")
log.info("=" * 60)

model_step = best_model["model"]
onehot_step = best_model["preprocessor"].named_transformers_["cat"]["onehot"]

# รวมชื่อฟีเจอร์ตัวเลขกับชื่อฟีเจอร์หมวดหมู่ที่ถูกแปลงด้วย One-Hot เรียบร้อยแล้ว
feature_names = numeric_features + list(onehot_step.get_feature_names_out(categorical_features))

# ดึงค่าความสำคัญ (Feature Importance) ตามเงื่อนไขประเภทสถาปัตยกรรมของโมเดลอย่างปลอดภัย
if hasattr(model_step, "feature_importances_"):
    importances = model_step.feature_importances_
    log.info("ดึงผลวิเคราะห์ด้วยคุณสมบัติ: .feature_importances_ (Tree-based Model)")
elif hasattr(model_step, "coef_"):
    # คำนวณค่าเฉลี่ยของน้ำหนักสัมบูรณ์ (axis=0) เพื่อให้รองรับกรณีโมเดลแบบ Multiclass ในอนาคต
    importances = np.mean(np.abs(model_step.coef_), axis=0)
    log.info("ดึงผลวิเคราะห์ด้วยคุณสมบัติ: |.coef_| (Linear Model)")
else:
    log.warning("โมเดลสถาปัตยกรรมนี้ไม่รองรับการดึงระดับความสำคัญของฟีเจอร์เชิงคณิตศาสตร์")
    importances = None

if importances is not None:
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)

    print("\n--- ตารางสรุปอันดับปัจจัยสำคัญต่อการทำนาย ---")
    print(importance_df.to_string(index=False))


# ============================================================
# STEP 9.5: ฟังก์ชันด้านความปลอดภัยและการเซฟโมเดลสำหรับงานจริง (Production Utilities)
# ============================================================
log.info("=" * 60)
log.info("STEP 9.5: สร้างเครื่องมือบันทึกโมเดลและระบบตรวจสอบข้อมูลขาเข้า")
log.info("=" * 60)

MODEL_PATH = Path("titanic_best_model.joblib")

def save_model(model, path: Path = MODEL_PATH) -> None:
    """บันทึก Pipeline ทั้งหมดลงไฟล์เก็บไว้ใช้งานทำนายในอนาคต"""
    joblib.dump(model, path)
    log.info(f"บันทึกไฟล์โมเดลไว้เรียบร้อยที่: {path.resolve()}")

def load_model(path: Path = MODEL_PATH):
    """โหลดไฟล์ Pipeline โมเดลกลับขึ้นมารันทำนายทันทีโดยไม่ต้องเทรนซ้ำ"""
    if not path.exists():
        raise FileNotFoundError(f"ไม่พบไฟล์บันทึกโมเดลที่ {path} กรุณาทำขั้นตอนฝึกโมเดลก่อนโหลดใช้งาน")
    return joblib.load(path)

# กำหนดกรอบโครงสร้างและข้อมูลนำเข้าที่ยอมรับได้
REQUIRED_COLUMNS = ["pclass", "sex", "age", "fare", "embarked"]
VALID_PCLASS = {1, 2, 3}
VALID_SEX = {"male", "female"}
VALID_EMBARKED = {"S", "C", "Q"}

def validate_passenger_input(df: pd.DataFrame) -> None:
    """ฟังก์ชันตรวจทานความถูกต้อง ความสมบูรณ์ และความถูกต้องของชนิดข้อมูลใหม่ (Input Validation)"""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"ข้อมูลขาเข้าต้องส่งในรูปแบบ DataFrame ของ pandas เท่านั้น | รูปแบบที่คุณส่งมา: {type(df).__name__}")

    # 1. เช็กว่าส่งคอลัมน์ที่จำเป็นมาครบหรือไม่
    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"ข้อมูลไม่ผ่านเกณฑ์ ขาดคอลัมน์สำคัญดังนี้: {sorted(missing_cols)}")

    # 2. เช็กว่าคอลัมน์ข้อมูลตัวเลขเป็นตัวเลขจริงๆ หรือไม่
    for col in ["pclass", "age", "fare"]:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise TypeError(f"ข้อมูลในคอลัมน์ '{col}' ต้องเป็นประเภทตัวเลขเท่านั้น")

    # 3. เช็กค่าข้อมูลที่ควรจะเป็นไปตามตัวเลือก
    bad_pclass = set(df["pclass"].dropna().unique()) - VALID_PCLASS
    if bad_pclass:
        raise ValueError(f"ค่าของ pclass ต้องเป็นเลข 1, 2 หรือ 3 เท่านั้น | ตรวจพบค่าผิดปกติ: {bad_pclass}")

    bad_sex = set(df["sex"].dropna().unique()) - VALID_SEX
    if bad_sex:
        raise ValueError(f"ค่าของ sex ต้องระบุเป็น male หรือ female เท่านั้น | ตรวจพบค่าผิดปกติ: {bad_sex}")

    bad_emb = set(df["embarked"].dropna().unique()) - VALID_EMBARKED
    if bad_emb:
        raise ValueError(f"ค่าของ embarked ต้องระบุเป็น S, C หรือ Q เท่านั้น | ตรวจพบค่าผิดปกติ: {bad_emb}")

    # 4. หากมีข้อมูลสูญหาย (Missing Value) แจ้งให้ระบบประมวลผลทราบล่วงหน้า
    if df[REQUIRED_COLUMNS].isnull().any().any():
        nulls = df[REQUIRED_COLUMNS].isnull().sum()
        log.info(f"⚠️ พบข้อมูลสูญหายในผู้โดยสารกลุ่มใหม่ — Imputer จะทำการแทนค่าด้วยค่าเฉลี่ย/มัธยฐานให้อัตโนมัติ:\n{nulls[nulls > 0].to_string()}")

# จัดการบันทึกท่อส่งโมเดลรุ่นที่ผ่านเกณฑ์ลงดิสก์
save_model(best_model)
log.info("ตัวแปรและโครงสร้างโมเดลถูกบันทึกเรียบร้อย สามารถนำไปประยุกต์ใช้งานหรือขึ้นคลาวด์ได้")


# ============================================================
# STEP 10: จำลองทำนายผลข้อมูลผู้โดยสารกลุ่มใหม่
# ============================================================
log.info("=" * 60)
log.info("STEP 10: สาธิตการทำนายผลลัพธ์ข้อมูลผู้โดยสารจำลอง")
log.info("=" * 60)

# จำลองตารางข้อมูลผู้โดยสารใหม่ 3 คน
new_passengers = pd.DataFrame({
    "pclass":   [1, 3, 2],
    "sex":      ["female", "male", "female"],
    "age":      [25, 30, 40],
    "fare":     [80.0, 7.5, 20.0],
    "embarked": ["C", "S", "Q"]
})

# รันระบบตรวจสอบข้อมูลขาเข้า
try:
    validate_passenger_input(new_passengers)
except (TypeError, ValueError) as e:
    log.error(f"ข้อมูลขัดข้องไม่สามารถรันระบบได้เนื่องจากคุณลักษณะนำเข้าขัดต่อกฎ: {e}")
    raise

# รันทำนายผลสัมฤทธิ์
predictions = best_model.predict(new_passengers)
probabilities = best_model.predict_proba(new_passengers)[:, 1]

print("\n--- รายชื่อตารางข้อมูลผู้โดยสารจำลองใหม่ ---")
print(new_passengers.to_string(index=False))

print("\n--- ผลลัพธ์วิเคราะห์และพยากรณ์จากโมเดล ---")
for i, (pred, prob) in enumerate(zip(predictions, probabilities), start=1):
    result = "รอดชีวิต (Survived)" if pred == 1 else "เสียชีวิต (Deceased)"
    log.info(f"  ผู้โดยสารคนย่อยที่ {i}: ผลทำนาย = {result:20s} (โอกาสรอดสูงสุดเฉลี่ย: {prob:.1%})")


# ============================================================
# ส่วนเสริม (Bonus): โหลดโมเดลกลับมาทำงานพยากรณ์โดยตรง
# ============================================================
log.info("=" * 60)
log.info("BONUS: โหลดไฟล์โมเดลสำเร็จรูปกลับเข้ามาใช้งาน")
log.info("=" * 60)

reloaded_model = load_model()
reloaded_preds = reloaded_model.predict(new_passengers)
log.info(f"ผลทำนายของผู้โดยสารใหม่จากโมเดลที่ดึงกลับขึ้นมา: {reloaded_preds.tolist()}")
log.info(f"การตรวจสอบความสอดคล้องเทียบเคียงผลลัพธ์เดิม: {reloaded_preds.tolist() == predictions.tolist()}")

log.info("=" * 60)
log.info("เสร็จสมบูรณ์! กระบวนการสร้างและประเมินผล ML Pipeline (v2) ดำเนินการเสร็จสมบูรณ์ครบถ้วนทุกขั้นตอน")
log.info("=" * 60)
