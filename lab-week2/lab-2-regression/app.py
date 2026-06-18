import logging
from pathlib import Path
import pandas as pd
import joblib
from flask import Flask, request, jsonify, render_template

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("penguins-web")

app = Flask(__name__)

# โหลดโมเดลสำหรับทำนายน้ำหนักตัวนกเพนกวิน
MODEL_PATH = Path("penguins_best_model.joblib")
if not MODEL_PATH.exists():
    log.error("ไม่พบไฟล์โมเดล penguins_best_model.joblib กรุณารันไฟล์ regression_pipeline.py ก่อน!")
    model = None
else:
    model = joblib.load(MODEL_PATH)
    log.info("โหลดโมเดลนกเพนกวินสำเร็จและพร้อมใช้งาน")

VALID_SPECIES = {"Adelie", "Chinstrap", "Gentoo"}
VALID_ISLANDS = {"Torgersen", "Biscoe", "Dream"}
VALID_SEX = {"male", "female"}

def validate_input(data):
    required = ["species", "island", "bill_length_mm", "bill_depth_mm", "flipper_length_mm", "sex"]
    for col in required:
        if col not in data:
            raise ValueError(f"กรุณากรอกข้อมูลให้ครบถ้วน ขาดช่อง: {col}")
            
    try:
        bill_len = float(data["bill_length_mm"])
        bill_depth = float(data["bill_depth_mm"])
        flipper_len = float(data["flipper_length_mm"])
    except (TypeError, ValueError):
        raise TypeError("ขนาดจงอยปากและขนาดครีบต้องเป็นตัวเลขเท่านั้น")

    if data["species"] not in VALID_SPECIES:
        raise ValueError("สายพันธุ์นกเพนกวินต้องเป็น Adelie, Chinstrap หรือ Gentoo")
    if data["island"] not in VALID_ISLANDS:
        raise ValueError("ชื่อเกาะต้องเป็น Torgersen, Biscoe หรือ Dream")
    if data["sex"] not in VALID_SEX:
        raise ValueError("เพศต้องเป็น male หรือ female")
    if bill_len <= 0 or bill_depth <= 0 or flipper_len <= 0:
        raise ValueError("ขนาดสัดส่วนของร่างกายห้ามมีค่าน้อยกว่าหรือเท่ากับศูนย์")
        
    return data["species"], data["island"], bill_len, bill_depth, flipper_len, data["sex"]

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
            
        species, island, bill_len, bill_depth, flipper_len, sex = validate_input(data)
        
        # แปลงข้อมูลให้อยู่ในรูป DataFrame สำหรับทำนาย
        input_df = pd.DataFrame({
            "species":           [species],
            "island":            [island],
            "bill_length_mm":    [bill_len],
            "bill_depth_mm":     [bill_depth],
            "flipper_length_mm": [flipper_len],
            "sex":               [sex]
        })
        
        # ทำนายน้ำหนัก
        predicted_mass = float(model.predict(input_df)[0])
        
        return jsonify({
            "success": True,
            "predicted_mass_g": predicted_mass,
            "predicted_mass_kg": predicted_mass / 1000
        })
        
    except (ValueError, TypeError) as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        log.error(f"เกิดข้อผิดพลาดที่คาดไม่ถึง: {e}")
        return jsonify({"success": False, "error": "เกิดข้อผิดพลาดภายในเซิร์ฟเวอร์"}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5056)
