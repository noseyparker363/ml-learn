import json

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 🐧 End-to-End Machine Learning Regression Pipeline\n",
    "### วิชา: CIT0013 Machine Learning | Lab 2: Penguin Body Mass Prediction (Regression)\n",
    "\n",
    "สมุดงาน (Notebook) นี้ถูกออกแบบมาสำหรับให้นักศึกษารันและเรียนรู้บน **Google Colab** เพื่อทำความเข้าใจขั้นตอนการสร้าง Machine Learning สำหรับทำนายค่าต่อเนื่องเชิงตัวเลข (Regression)\n",
    "\n",
    "#### 🌟 เปรียบเทียบความแตกต่างสำคัญกับ Lab 1 (Titanic Classification):\n",
    "1. **ตัวแปรเป้าหมาย (Target - y):** รอบนี้คือน้ำหนักของนกเพนกวิน (`body_mass_g`) ซึ่งเป็นตัวเลขเชิงปริมาณที่มีความต่อเนื่อง (ไม่ใช่ 0 หรือ 1)\n",
    "2. **การแบ่งกลุ่มข้อมูล (Train/Test Split):** ในงาน Regression เราไม่จำเป็นต้องใช้พารามิเตอร์ `stratify` ในการแบ่งกลุ่มข้อมูล\n",
    "3. **เกณฑ์การวัดผล (Metrics):** เราจะหันมาวัดผลด้วย **MAE** (ความคลาดเคลื่อนเฉลี่ย), **RMSE** (ส่วนเบี่ยงเบนความคลาดเคลื่อน) และ **R-squared** (อัตราการอธิบายความผันแปรของข้อมูล)\n",
    "4. **โมเดลที่เปรียบเทียบ:** จะใช้สถาปัตยกรรมกลุ่ม Regressor ได้แก่ Linear Regression, Decision Tree Regressor และ Random Forest Regressor"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# ติดตั้งไลบรารีที่จำเป็นสำหรับ Google Colab\n",
    "!pip install seaborn pandas scikit-learn numpy joblib"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "## 🛠️ Step 1: นำเข้า Libraries และตั้งค่า Logging\n",
    "นำเข้าเครื่องมือฝั่ง Regressor และเครื่องมือวัดผลสำหรับประเมินค่าความคลาดเคลื่อนทางคณิตศาสตร์"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import seaborn as sns\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import logging\n",
    "import joblib\n",
    "from pathlib import Path\n",
    "\n",
    "from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV\n",
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.compose import ColumnTransformer\n",
    "from sklearn.impute import SimpleImputer\n",
    "from sklearn.preprocessing import StandardScaler, OneHotEncoder\n",
    "from sklearn.linear_model import LinearRegression\n",
    "from sklearn.tree import DecisionTreeRegressor\n",
    "from sklearn.ensemble import RandomForestRegressor\n",
    "from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score\n",
    "\n",
    "import warnings\n",
    "warnings.filterwarnings(\"ignore\")\n",
    "\n",
    "logging.basicConfig(\n",
    "    level=logging.INFO,\n",
    "    format=\"%(asctime)s | %(levelname)-7s | %(message)s\",\n",
    "    datefmt=\"%H:%M:%S\"\n",
    ")\n",
    "log = logging.getLogger(\"penguins-ml\")\n",
    "\n",
    "for noisy in (\"matplotlib\", \"seaborn\"):\n",
    "    logging.getLogger(noisy).setLevel(logging.WARNING)\n",
    "\n",
    "log.info(\"Step 1: โหลด Libraries และตั้งค่าเสร็จสิ้น\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "## 📥 Step 2: โหลดข้อมูลนกเพนกวิน (Penguins Dataset)\n",
    "ข้อมูลนกเพนกวิน 3 สายพันธุ์จากหมู่เกาะ Palmer Archipelago ในทวีปแอนตาร์กติกา"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "penguins = sns.load_dataset(\"penguins\")\n",
    "\n",
    "log.info(f\"Step 2: โหลดข้อมูลนกเพนกวินสำเร็จ ขนาดข้อมูล={penguins.shape}\")\n",
    "print(\"\\n--- ตัวอย่างข้อมูล 5 แถวแรก ---\")\n",
    "display(penguins.head())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "## 🔍 Step 3: ตรวจสอบข้อมูลเบื้องต้นและค่าสูญหาย (Quick Look)\n",
    "วิเคราะห์ลักษณะการกระจายตัวของน้ำหนักนกเพนกวิน และดูว่ามีข้อมูลช่องใดหายไปบ้าง"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "log.info(f\"น้ำหนักเฉลี่ยของนกเพนกวิน: {penguins['body_mass_g'].mean():.2f} กรัม\")\n",
    "log.info(f\"น้ำหนักต่ำสุด: {penguins['body_mass_g'].min()} กรัม | สูงสุด: {penguins['body_mass_g'].max()} กรัม\")\n",
    "\n",
    "print(\"\\n--- ตรวจสอบจำนวนข้อมูลสูญหาย ---\")\n",
    "missing_summary = penguins.isnull().sum()\n",
    "print(missing_summary[missing_summary > 0])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "## ✂️ Step 4: เลือกคุณลักษณะ (Features) และแบ่งกลุ่ม Train/Test Set\n",
    "แยกข้อมูลชุดสอน (80%) และชุดทดสอบ (20%)\n",
    "\n",
    "*หมายเหตุสำคัญ: สำหรับ Regression เราจะไม่ใช้ `stratify=y` เหมือนใน Classification เนื่องจากตัวแปรเป้าหมายเป็นจำนวนตัวเลขแบบต่อเนื่องไม่ใช่ระดับกลุ่มคลาส*"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "feature_cols = [\"species\", \"island\", \"bill_length_mm\", \"bill_depth_mm\", \"flipper_length_mm\", \"sex\"]\n",
    "\n",
    "# ลบแถวที่ค่าน้ำหนักเป้าหมายขาดหายออกก่อน\n",
    "penguins_clean = penguins.dropna(subset=[\"body_mass_g\"]).copy()\n",
    "\n",
    "X = penguins_clean[feature_cols]\n",
    "y = penguins_clean[\"body_mass_g\"]\n",
    "\n",
    "# แบ่งชุดข้อมูล\n",
    "X_train, X_test, y_train, y_test = train_test_split(\n",
    "    X, y, test_size=0.2, random_state=42\n",
    ")\n",
    "\n",
    "log.info(f\"ขนาดชุดเรียนรู้ (Train Set): {X_train.shape[0]} แถว | ขนาดชุดประเมิน (Test Set): {X_test.shape[0]} แถว\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "## 🤖 Step 5: สร้างท่อส่งข้อมูล Preprocessing Pipeline\n",
    "จัดการประเภทข้อมูลก่อนส่งให้โมเดลประมวลผล:\n",
    "1. **ข้อมูลตัวเลข (Numeric features):** bill_length_mm, bill_depth_mm, flipper_length_mm (เติมข้อมูลด้วยค่า Median และทำ StandardScaler)\n",
    "2. **ข้อมูลหมวดหมู่ (Categorical features):** species, island, sex (เติมข้อมูลสูญหายด้วยฐานนิยม และทำ One-Hot Encoding)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "numeric_features = [\"bill_length_mm\", \"bill_depth_mm\", \"flipper_length_mm\"]\n",
    "categorical_features = [\"species\", \"island\", \"sex\"]\n",
    "\n",
    "# 1) ท่อสำหรับกลุ่มตัวเลข\n",
    "numeric_transformer = Pipeline(steps=[\n",
    "    (\"imputer\", SimpleImputer(strategy=\"median\")),\n",
    "    (\"scaler\", StandardScaler())\n",
    "])\n",
    "\n",
    "# 2) ท่อสำหรับกลุ่มข้อมูลตัวอักษร\n",
    "categorical_transformer = Pipeline(steps=[\n",
    "    (\"imputer\", SimpleImputer(strategy=\"most_frequent\")),\n",
    "    (\"onehot\", OneHotEncoder(handle_unknown=\"ignore\", sparse_output=False))\n",
    "])\n",
    "\n",
    "# 3) รวมเข้าด้วยกันเป็น Preprocessor หลัก\n",
    "preprocessor = ColumnTransformer(transformers=[\n",
    "    (\"num\", numeric_transformer, numeric_features),\n",
    "    (\"cat\", categorical_transformer, categorical_features)\n",
    "])\n",
    "\n",
    "log.info(\"สร้าง Preprocessing Pipeline เสร็จเรียบร้อย\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "## 📊 Step 6: เปรียบเทียบประสิทธิภาพโมเดลด้วย Cross-Validation\n",
    "เราเปรียบเทียบโมเดลทำนายตัวเลข 3 ตัว: Linear Regression, Decision Tree Regressor, และ Random Forest Regressor\n",
    "\n",
    "*เกณฑ์ประเมิน:* เราวัดความสามารถเบื้องต้นด้วย **Mean Absolute Error (MAE)** ซึ่งบ่งบอกความคลาดเคลื่อนเฉลี่ยว่าโมเดลพยากรณ์คลาดเคลื่อนไปกี่กรัม (ค่ายิ่งน้อยยิ่งดี)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "models = {\n",
    "    \"Linear Regression\":       LinearRegression(),\n",
    "    \"Decision Tree Regressor\": DecisionTreeRegressor(random_state=42),\n",
    "    \"Random Forest Regressor\": RandomForestRegressor(n_estimators=100, random_state=42)\n",
    "}\n",
    "\n",
    "results = {}\n",
    "for name, model in models.items():\n",
    "    pipe = Pipeline(steps=[\n",
    "        (\"preprocessor\", preprocessor),\n",
    "        (\"model\", model)\n",
    "    ])\n",
    "    # ใช้ scoring='neg_mean_absolute_error' (ได้คะแนนเฉลี่ยเป็นค่าติดลบ)\n",
    "    scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring=\"neg_mean_absolute_error\")\n",
    "    # แปลงเป็นค่าบวกเพื่อให้เข้าใจง่าย (MAE)\n",
    "    mae_scores = -scores\n",
    "    results[name] = mae_scores\n",
    "    log.info(f\"  {name:25s}: MAE เฉลี่ย = {mae_scores.mean():.2f} กรัม (+/- ส่วนเบี่ยงเบน {mae_scores.std():.2f})\")\n",
    "\n",
    "# เลือกโมเดลที่คลาดเคลื่อนน้อยที่สุดโดยอัตโนมัติ\n",
    "best_model_name = min(results, key=lambda k: results[k].mean())\n",
    "best_mae = results[best_model_name].mean()\n",
    "log.info(f\"⭐ โมเดลที่ดีที่สุดจากการรัน CV: {best_model_name} (MAE ต่ำที่สุดเฉลี่ย = {best_mae:.2f} กรัม)\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "## ⚙️ Step 7: ปรับพารามิเตอร์แบบไดนามิกด้วย GridSearchCV\n",
    "ค้นหาพารามิเตอร์ที่ดีที่สุดของโมเดลตัวที่ให้ผลลัพธ์คลาดเคลื่อนน้อยที่สุดจากการสแกนเบื้องต้น"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "param_grids = {\n",
    "    \"Linear Regression\": {\n",
    "        \"model__fit_intercept\": [True, False]\n",
    "    },\n",
    "    \"Decision Tree Regressor\": {\n",
    "        \"model__max_depth\":         [None, 5, 10],\n",
    "        \"model__min_samples_split\": [2, 5, 10]\n",
    "    },\n",
    "    \"Random Forest Regressor\": {\n",
    "        \"model__n_estimators\":      [50, 100, 150],\n",
    "        \"model__max_depth\":         [None, 5, 10],\n",
    "        \"model__min_samples_split\": [2, 5]\n",
    "    }\n",
    "}\n",
    "\n",
    "best_model_class = type(models[best_model_name])\n",
    "selected_grid = param_grids.get(best_model_name)\n",
    "\n",
    "if selected_grid is None:\n",
    "    selected_grid = param_grids[\"Linear Regression\"]\n",
    "    best_model_class = LinearRegression\n",
    "\n",
    "log.info(f\"สถาปัตยกรรมที่นำมาจูน: {best_model_name}\")\n",
    "\n",
    "final_pipe = Pipeline(steps=[\n",
    "    (\"preprocessor\", preprocessor),\n",
    "    (\"model\", best_model_class())\n",
    "])\n",
    "\n",
    "grid_search = GridSearchCV(\n",
    "    final_pipe, selected_grid, cv=5,\n",
    "    scoring=\"neg_mean_absolute_error\", n_jobs=-1\n",
    ")\n",
    "grid_search.fit(X_train, y_train)\n",
    "\n",
    "log.info(f\"Best parameters: {grid_search.best_params_}\")\n",
    "log.info(f\"Best CV MAE: {-grid_search.best_score_:.2f} กรัม\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "## 🎯 Step 8: ประเมินผลประสิทธิภาพโมเดลบนชุดข้อมูลทดสอบ (Test Set)\n",
    "ทำการวัดผลผ่านตัววัดสำคัญ 3 ตัวเพื่อสรุปเป็นความแม่นยำของระบบพยากรณ์ค่าน้ำหนักตัว:\n",
    "1. **Mean Absolute Error (MAE):** ความคลาดเคลื่อนสัมบูรณ์เฉลี่ย\n",
    "2. **Root Mean Squared Error (RMSE):** ความคลาดเคลื่อนรากที่สองเฉลี่ย\n",
    "3. **R-squared (R2 Score):** สัมประสิทธิ์การตัดสินใจ (ยิ่งเข้าใกล้ 1.0 หรือ 100% ยิ่งสามารถทำนายอธิบายการขึ้นลงของน้ำหนักจริงได้ครอบคลุม)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "best_model = grid_search.best_estimator_\n",
    "y_pred = best_model.predict(X_test)\n",
    "\n",
    "mae = mean_absolute_error(y_test, y_pred)\n",
    "mse = mean_squared_error(y_test, y_pred)\n",
    "rmse = np.sqrt(mse)\n",
    "r2 = r2_score(y_test, y_pred)\n",
    "\n",
    "print(\"\\n--- รายงานประเมินผลลัพธ์โมเดล Regression (ชุดทดสอบ) ---\")\n",
    "print(f\"  ค่าเฉลี่ยความคลาดเคลื่อนสัมบูรณ์ (MAE):   {mae:.2f} กรัม\")\n",
    "print(f\"  ส่วนเบี่ยงเบนความคลาดเคลื่อน (RMSE):     {rmse:.2f} กรัม\")\n",
    "print(f\"  ค่าอธิบายความผันแปร R-squared (R2 Score): {r2:.3f} (หรือประมาณ {r2:.1%})\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "## 💡 Step 9: ถอดรหัสระดับอิทธิพลปัจจัยสำคัญ (Feature Importance / Coefficients)\n",
    "สแกนเพื่อตรวจสอบว่าฟีเจอร์ใดเป็นตัวแปรเพิ่มหรือลดค่าน้ำหนักตัวเพนกวินมากที่สุด"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "model_step = best_model[\"model\"]\n",
    "onehot_step = best_model[\"preprocessor\"].named_transformers_[\"cat\"][\"onehot\"]\n",
    "feature_names = numeric_features + list(onehot_step.get_feature_names_out(categorical_features))\n",
    "\n",
    "if hasattr(model_step, \"feature_importances_\"):\n",
    "    importances = model_step.feature_importances_\n",
    "    log.info(\"ถอดรหัสความสำคัญด้วย: .feature_importances_ (Tree-based)\")\n",
    "elif hasattr(model_step, \"coef_\"):\n",
    "    importances = model_step.coef_\n",
    "    log.info(\"ถอดรหัสความสำคัญด้วยน้ำหนักสัมประสิทธิ์เชิงเส้น: .coef_ (Linear Regression)\")\n",
    "else:\n",
    "    importances = None\n",
    "\n",
    "if importances is not None:\n",
    "    importance_df = pd.DataFrame({\n",
    "        \"feature\": feature_names,\n",
    "        \"value\": importances\n",
    "    })\n",
    "    \n",
    "    # จัดเรียงระดับอิทธิพลโดยพิจารณาจากขนาดสมบูรณ์ (กรณีเชิงเส้น)\n",
    "    if hasattr(model_step, \"coef_\"):\n",
    "        importance_df[\"abs_val\"] = importance_df[\"value\"].abs()\n",
    "        importance_df = importance_df.sort_values(\"abs_val\", ascending=False).drop(columns=[\"abs_val\"])\n",
    "    else:\n",
    "        importance_df = importance_df.sort_values(\"value\", ascending=False)\n",
    "\n",
    "    print(\"\\n--- อันดับคุณสมบัติที่มีอิทธิพลต่อน้ำหนักตัว ---\")\n",
    "    print(importance_df.to_string(index=False))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "## 📦 Step 9.5 & 10: เซฟตัวแปรโมเดลและการรันทำนายค่าข้อมูลใหม่\n",
    "จำลองข้อมูลนกเพนกวินตัวใหม่ส่งให้ระบบเช็กโครงสร้างและรันทำนายค่าน้ำหนักต่อเนื่องแบบกรัม"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# บันทึกโมเดลหลักลงคลาวด์/ดิสก์\n",
    "MODEL_PATH = Path(\"penguins_best_model.joblib\")\n",
    "joblib.dump(best_model, MODEL_PATH)\n",
    "log.info(f\"เซฟโมเดลสำเร็จแล้วที่: {MODEL_PATH.resolve()}\")\n",
    "\n",
    "# ข้อมูลจำลองนกเพนกวินตัวใหม่\n",
    "new_penguins = pd.DataFrame({\n",
    "    \"species\":           [\"Adelie\", \"Gentoo\"],\n",
    "    \"island\":            [\"Torgersen\", \"Biscoe\"],\n",
    "    \"bill_length_mm\":    [39.1, 48.2],\n",
    "    \"bill_depth_mm\":     [18.7, 14.3],\n",
    "    \"flipper_length_mm\": [181.0, 210.0],\n",
    "    \"sex\":               [\"male\", \"female\"]\n",
    "})\n",
    "\n",
    "# รันทำนายค่าน้ำหนัก\n",
    "predicted_mass = best_model.predict(new_penguins)\n",
    "\n",
    "print(\"\\n--- รายชื่อข้อมูลนกเพนกวินจำลองตัวใหม่ ---\")\n",
    "display(new_penguins)\n",
    "\n",
    "print(\"\\n--- สรุปค่าน้ำหนักที่พยากรณ์ได้ ---\")\n",
    "for i, pred in enumerate(predicted_mass, start=1):\n",
    "    log.info(f\"นกเพนกวินตัวที่ {i}: ทำนายน้ำหนักได้ = {pred:.2f} กรัม (ประมาณ {pred/1000:.2f} กิโลกรัม)\")\n",
    "\n",
    "log.info(\"🎉 สิ้นสุดกระบวนการทดสอบ ML Regression Pipeline สำเร็จครบถ้วน\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

with open("regression_pipeline.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print("Notebook regression_pipeline.ipynb created successfully.")
