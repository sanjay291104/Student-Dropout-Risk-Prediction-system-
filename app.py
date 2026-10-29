"""
Student Dropout Risk Prediction and Intervention Recommendation System
Flask application (app.py)
"""
import json
import os
from datetime import datetime

import joblib
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
app.config["SECRET_KEY"] = "dropout-prediction-secret-key-2026"
app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{os.path.join(BASE_DIR, 'database.db')}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# ---------------------------------------------------------------------------
# Load ML artifacts
# ---------------------------------------------------------------------------
MODEL_DIR = os.path.join(BASE_DIR, "models")
best_model = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
encoders = joblib.load(os.path.join(MODEL_DIR, "encoders.pkl"))
feature_columns = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))

with open(os.path.join(MODEL_DIR, "metrics.json")) as f:
    METRICS = json.load(f)

with open(os.path.join(MODEL_DIR, "feature_importance.json")) as f:
    FEATURE_IMPORTANCE = json.load(f)

CATEGORICAL_COLS = ["Gender", "Internet_Access", "Part_Time_Job", "Scholarship",
                     "Semester", "Department", "Parental_Education"]
NUMERIC_COLS = ["Age", "Family_Income", "Study_Hours_per_Day", "Attendance_Rate",
                 "Assignment_Delay_Days", "Travel_Time_Minutes", "Stress_Index",
                 "GPA", "Semester_GPA", "CGPA"]


# ---------------------------------------------------------------------------
# Database Models
# ---------------------------------------------------------------------------
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    institution = db.Column(db.String(150))
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def set_password(self, pw):
        self.password_hash = generate_password_hash(pw)

    def check_password(self, pw):
        return check_password_hash(self.password_hash, pw)


class PredictionHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    student_name = db.Column(db.String(120))
    input_data = db.Column(db.Text)
    risk_level = db.Column(db.String(20))
    probability = db.Column(db.Float)
    recommendations = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


with app.app_context():
    db.create_all()


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def login_required(view_func):
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in to continue.", "warning")
            return redirect(url_for("login"))
        return view_func(*args, **kwargs)
    wrapper.__name__ = view_func.__name__
    return wrapper


def get_risk_level(prob):
    if prob >= 0.66:
        return "High"
    elif prob >= 0.33:
        return "Medium"
    return "Low"


def generate_recommendations(data, prob):
    """Rule-based explainable intervention recommendations."""
    recs = []

    if data["Attendance_Rate"] < 75:
        recs.append({
            "title": "Attendance Improvement Plan",
            "detail": "Attendance is below 75%. Enroll the student in mandatory "
                       "attendance monitoring with weekly mentor check-ins.",
            "icon": "calendar"
        })
    if data["CGPA"] < 2.0 or data["Semester_GPA"] < 2.0:
        recs.append({
            "title": "Academic Counseling & Remedial Classes",
            "detail": "Current GPA indicates academic struggle. Recommend remedial "
                       "classes and one-on-one faculty counseling sessions.",
            "icon": "book"
        })
    if data["Assignment_Delay_Days"] >= 3:
        recs.append({
            "title": "Time Management Mentoring",
            "detail": "Frequent assignment delays detected. Pair the student with a "
                       "peer mentor and set up a structured submission tracker.",
            "icon": "clock"
        })
    if data["Stress_Index"] >= 7:
        recs.append({
            "title": "Wellness & Counseling Support",
            "detail": "Elevated stress levels detected. Refer the student to the "
                       "institution's counseling and wellness center.",
            "icon": "heart"
        })
    if data["Family_Income"] < 20000:
        recs.append({
            "title": "Financial Aid Guidance",
            "detail": "Household income suggests financial strain. Guide the student "
                       "toward scholarships, fee waivers, or education loans.",
            "icon": "money"
        })
    if data["Part_Time_Job"] == "Yes" and data["Study_Hours_per_Day"] < 2:
        recs.append({
            "title": "Workload Balancing Plan",
            "detail": "Part-time job combined with low study hours raises risk. "
                       "Help the student restructure their weekly schedule.",
            "icon": "balance"
        })
    if data["Travel_Time_Minutes"] > 60:
        recs.append({
            "title": "Accommodation / Hostel Assistance",
            "detail": "Long commute time may impact attendance and energy. "
                       "Consider hostel or nearby accommodation support.",
            "icon": "map"
        })

    if not recs:
        recs.append({
            "title": "Continue Regular Monitoring",
            "detail": "No major risk factors detected. Maintain routine academic "
                       "monitoring and periodic check-ins.",
            "icon": "check"
        })

    return recs


def preprocess_input(form):
    """Convert raw form input into model-ready feature vector."""
    data = {
        "Gender": form["Gender"],
        "Internet_Access": form["Internet_Access"],
        "Part_Time_Job": form["Part_Time_Job"],
        "Scholarship": form["Scholarship"],
        "Semester": form["Semester"],
        "Department": form["Department"],
        "Parental_Education": form["Parental_Education"],
        "Age": float(form["Age"]),
        "Family_Income": float(form["Family_Income"]),
        "Study_Hours_per_Day": float(form["Study_Hours_per_Day"]),
        "Attendance_Rate": float(form["Attendance_Rate"]),
        "Assignment_Delay_Days": float(form["Assignment_Delay_Days"]),
        "Travel_Time_Minutes": float(form["Travel_Time_Minutes"]),
        "Stress_Index": float(form["Stress_Index"]),
        "GPA": float(form["GPA"]),
        "Semester_GPA": float(form["Semester_GPA"]),
        "CGPA": float(form["CGPA"]),
    }

    encoded = {}
    for col in CATEGORICAL_COLS:
        le = encoders[col]
        val = data[col] if data[col] in le.classes_ else le.classes_[0]
        encoded[col] = le.transform([val])[0]
    for col in NUMERIC_COLS:
        encoded[col] = data[col]

    row = [encoded[c] for c in feature_columns]
    row_scaled = scaler.transform([row])
    return row_scaled, data


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def landing():
    return render_template("landing.html")


@app.route("/about")
def about():
    return render_template("about.html", metrics=METRICS)


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        full_name = request.form.get("full_name", "").strip()
        email = request.form.get("email", "").strip().lower()
        institution = request.form.get("institution", "").strip()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm_password", "")

        if not full_name or not email or not password:
            flash("Please fill in all required fields.", "error")
            return redirect(url_for("register"))
        if password != confirm:
            flash("Passwords do not match.", "error")
            return redirect(url_for("register"))
        if User.query.filter_by(email=email).first():
            flash("An account with this email already exists.", "error")
            return redirect(url_for("register"))

        user = User(full_name=full_name, email=email, institution=institution)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()

        flash("Account created successfully! Please log in.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        user = User.query.filter_by(email=email).first()

        if user and user.check_password(password):
            session["user_id"] = user.id
            session["full_name"] = user.full_name
            flash(f"Welcome back, {user.full_name}!", "success")
            return redirect(url_for("dashboard"))
        flash("Invalid email or password.", "error")
        return redirect(url_for("login"))

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "success")
    return redirect(url_for("landing"))


@app.route("/dashboard", methods=["GET", "POST"])
@login_required
def dashboard():
    result = None
    if request.method == "POST":
        try:
            row_scaled, raw_data = preprocess_input(request.form)
            prob = float(best_model.predict_proba(row_scaled)[0][1])
            risk_level = get_risk_level(prob)
            recommendations = generate_recommendations(raw_data, prob)

            history = PredictionHistory(
                user_id=session["user_id"],
                student_name=request.form.get("student_name", "Unnamed Student"),
                input_data=json.dumps(raw_data),
                risk_level=risk_level,
                probability=round(prob * 100, 2),
                recommendations=json.dumps(recommendations),
            )
            db.session.add(history)
            db.session.commit()

            result = {
                "student_name": request.form.get("student_name", "Unnamed Student"),
                "probability": round(prob * 100, 2),
                "risk_level": risk_level,
                "recommendations": recommendations,
                "raw_data": raw_data,
            }
        except Exception as e:
            flash(f"Error processing prediction: {e}", "error")

    return render_template(
        "dashboard.html",
        result=result,
        metrics=METRICS,
        feature_importance=FEATURE_IMPORTANCE,
        best_model_name=METRICS.get("best_model", "XGBoost"),
    )


@app.route("/history")
@login_required
def history():
    records = (PredictionHistory.query
               .filter_by(user_id=session["user_id"])
               .order_by(PredictionHistory.created_at.desc())
               .all())
    parsed = []
    for r in records:
        parsed.append({
            "id": r.id,
            "student_name": r.student_name,
            "risk_level": r.risk_level,
            "probability": r.probability,
            "recommendations": json.loads(r.recommendations),
            "input_data": json.loads(r.input_data),
            "created_at": r.created_at.strftime("%d %b %Y, %I:%M %p"),
        })
    return render_template("history.html", records=parsed)


@app.route("/history/delete/<int:record_id>", methods=["POST"])
@login_required
def delete_history(record_id):
    record = PredictionHistory.query.filter_by(id=record_id, user_id=session["user_id"]).first()
    if record:
        db.session.delete(record)
        db.session.commit()
        flash("Record deleted.", "success")
    return redirect(url_for("history"))


@app.route("/api/metrics")
def api_metrics():
    return jsonify(METRICS)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
