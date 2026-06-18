import logging
from pathlib import Path
import pandas as pd
import joblib
from flask import Flask, request, jsonify, render_template

# ตั้งค่า Logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("titanic-web")

app = Flask(__name__)

# โหลดโมเดลที่บันทึกไว้
MODEL_PATH = Path("titanic_best_model.joblib")
if not MODEL_PATH.exists():
    log.error("ไม่พบไฟล์โมเดล titanic_best_model.joblib กรุณารันไฟล์ ml_pipeline_v2.py เพื่อเตรียมโมเดลก่อน!")
    model = None
else:
    model = joblib.load(MODEL_PATH)
    log.info("โหลดโมเดลสำเร็จและพร้อมใช้งาน")

# ข้อจำกัดข้อมูลในการตรวจสอบ
VALID_PCLASS = {1, 2, 3}
VALID_SEX = {"male", "female"}
VALID_EMBARKED = {"S", "C", "Q"}

def validate_input(data):
    """ฟังก์ชันสกรีนข้อมูลฝั่งเซิร์ฟเวอร์ก่อนส่งเข้าโมเดล"""
    required = ["pclass", "sex", "age", "fare", "embarked"]
    for col in required:
        if col not in data:
            raise ValueError(f"กรุณากรอกข้อมูลให้ครบถ้วน ขาดช่อง: {col}")
            
    # ตรวจสอบประเภทและค่าของแต่ละฟีเจอร์
    try:
        pclass = int(data["pclass"])
        age = float(data["age"])
        fare = float(data["fare"])
    except (TypeError, ValueError):
        raise TypeError("คอลัมน์ pclass, age, และ fare ต้องเป็นตัวเลขเท่านั้น")

    if pclass not in VALID_PCLASS:
        raise ValueError("ระดับชั้นตั๋ว (pclass) ต้องเป็น 1, 2 หรือ 3 เท่านั้น")
    if data["sex"] not in VALID_SEX:
        raise ValueError("เพศ (sex) ต้องเป็น male หรือ female เท่านั้น")
    if data["embarked"] not in VALID_EMBARKED:
        raise ValueError("ท่าเรือ (embarked) ต้องเป็น S, C หรือ Q เท่านั้น")
    if age < 0 or fare < 0:
        raise ValueError("อายุและค่าตั๋วห้ามมีค่าติดลบ")
        
    return pclass, data["sex"], age, fare, data["embarked"]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "โมเดลยังไม่พร้อมใช้งานบนเซิร์ฟเวอร์"}), 500
        
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "ไม่พบข้อมูลที่ส่งเข้ามา"}), 400
            
        # ตรวจสอบความสมบูรณ์ข้อมูล
        pclass, sex, age, fare, embarked = validate_input(data)
        
        # จัดแปลงข้อมูลให้อยู่ในรูป DataFrame เหมือนตอนฝึกฝนโมเดล
        input_df = pd.DataFrame({
            "pclass":   [pclass],
            "sex":      [sex],
            "age":      [age],
            "fare":     [fare],
            "embarked": [embarked]
        })
        
        # รันทำนาย
        prediction = int(model.predict(input_df)[0])
        probability = float(model.predict_proba(input_df)[0, 1])
        
        return jsonify({
            "success": True,
            "prediction": prediction,
            "probability": probability,
            "message": "รอดชีวิต" if prediction == 1 else "เสียชีวิต"
        })
        
    except (ValueError, TypeError) as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        log.error(f"เกิดข้อผิดพลาดที่คาดไม่ถึง: {e}")
        return jsonify({"success": False, "error": "เกิดข้อผิดพลาดภายในเซิร์ฟเวอร์"}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5055)
