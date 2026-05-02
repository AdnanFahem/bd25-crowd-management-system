# =========================================================
# EVENT MANAGEMENT DASHBOARD - FINAL BACKEND
# =========================================================
# Features:
# 1. FastAPI backend
# 2. SQLite database
# 3. Users, events, sessions, footfall_data, survey_data
# 4. Password hashing with bcrypt
# 5. Role-based login (admin, staff, viewer)
# 6. Admin user management
# 7. Event CRUD with role restrictions
# 8. ML prediction using real HUQ footfall data
# 9. Survey analytics
# 10. Real client event-like records from footfall data
# 11. Smart client event search with ML-assisted results
# =========================================================

from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sqlite3
import pandas as pd
from sklearn.linear_model import LinearRegression
import bcrypt
import os
import secrets
from typing import Optional
import re

# =========================================================
# DATABASE SETTINGS
# =========================================================

DB_NAME = "events.db"
FOOTFALL_CSV = "footfall_data.csv"
SURVEY_FILE = "survey_data.xlsx"

FOOTFALL_THRESHOLDS = {
    "low": 1000.0,
    "medium": 5000.0
}


def get_db_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn


# =========================================================
# PASSWORD HASHING
# =========================================================

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), hashed_password.encode("utf-8"))


# =========================================================
# DATABASE SETUP
# =========================================================

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            date TEXT NOT NULL,
            location TEXT NOT NULL,
            footfall INTEGER NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            token TEXT UNIQUE NOT NULL,
            username TEXT NOT NULL,
            role TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


# =========================================================
# SEED DEFAULT USERS
# =========================================================
# Add more users here if needed:
# ("newuser", hash_password("newpassword"), "staff")
# roles: admin, staff, viewer
# =========================================================

def seed_users():
    conn = get_db_connection()
    cursor = conn.cursor()

    users = [
        ("admin", hash_password("admin123"), "admin"),
        ("staff1", hash_password("staff123"), "staff"),
        ("viewer1", hash_password("viewer123"), "viewer"),
    ]

    for user in users:
        try:
            cursor.execute(
                "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
                user
            )
        except sqlite3.IntegrityError:
            pass

    conn.commit()
    conn.close()


# =========================================================
# SEED DEFAULT EVENTS
# =========================================================
# No fake demo events.
# Admin/staff can create managed events from the UI.
# =========================================================

def seed_events():
    return


# =========================================================
# IMPORT REAL HUQ FOOTFALL DATA INTO SQLITE
# =========================================================

def import_footfall_data():
    if not os.path.exists(FOOTFALL_CSV):
        print("HUQ data file not found:", FOOTFALL_CSV)
        return

    df = pd.read_csv(FOOTFALL_CSV)
    df.columns = df.columns.str.strip()

    conn = get_db_connection()
    df.to_sql("footfall_data", conn, if_exists="replace", index=False)
    conn.close()

    print("HUQ footfall data imported into SQLite table: footfall_data")


# =========================================================
# SURVEY DATA HELPERS
# =========================================================

def make_unique_columns(columns):
    seen = {}
    unique_columns = []

    for col in columns:
        name = str(col).strip() if col is not None else "Unnamed"
        if name == "":
            name = "Unnamed"

        if name not in seen:
            seen[name] = 1
            unique_columns.append(name)
        else:
            seen[name] += 1
            unique_columns.append(f"{name}_{seen[name]}")

    return unique_columns


def choose_best_survey_sheet(all_sheets: dict) -> tuple[str, pd.DataFrame]:
    """
    Choose the sheet that looks most like the real survey responses.
    Preference goes to sheets with:
    - more rows
    - more non-empty cells
    - more columns containing survey keywords
    """
    best_sheet_name = None
    best_df = None
    best_score = -1

    keywords = [
        "category",
        "event",
        "venue",
        "attended",
        "screening",
        "message",
        "accessibility",
        "facilities",
        "location",
        "layout",
        "food",
        "drink",
        "suggestion"
    ]

    for sheet_name, df in all_sheets.items():
        temp_df = df.copy()
        temp_df.columns = make_unique_columns(temp_df.columns)
        temp_df = temp_df.dropna(how="all")
        temp_df = temp_df.dropna(axis=1, how="all")

        non_empty_cells = int(temp_df.notna().sum().sum())
        row_count = len(temp_df)

        keyword_matches = 0
        for col in temp_df.columns:
            col_lower = str(col).lower()
            if any(keyword in col_lower for keyword in keywords):
                keyword_matches += 1

        score = (row_count * 10) + non_empty_cells + (keyword_matches * 100)

        if score > best_score:
            best_score = score
            best_sheet_name = sheet_name
            best_df = temp_df

    return best_sheet_name, best_df


def clean_survey_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = make_unique_columns(df.columns)

    # Remove fully empty rows and columns
    df = df.dropna(how="all")
    df = df.dropna(axis=1, how="all")

    # Convert obvious Excel errors to NaN-like blanks
    for col in df.columns:
        df[col] = df[col].replace(
            ["#REF!", "#N/A", "#VALUE!", "#DIV/0!", "nan", "None"],
            pd.NA
        )

    return df


def is_meaningful_survey_value(value) -> bool:
    if value is None:
        return False

    text = str(value).strip()

    if text == "":
        return False

    junk_values = {
        "nan", "none", "null", "#ref!", "#n/a", "#value!", "#div/0!",
        "0", "0.0", "1", "1.0", "true", "false"
    }

    if text.lower() in junk_values:
        return False

    # Ignore pure numeric values
    if re.fullmatch(r"\d+(\.\d+)?", text):
        return False

    return True


def count_meaningful_values(series: pd.Series) -> dict:
    cleaned = []
    for value in series.dropna():
        if is_meaningful_survey_value(value):
            cleaned.append(str(value).strip())

    if not cleaned:
        return {}

    return pd.Series(cleaned).value_counts().to_dict()


# =========================================================
# IMPORT SURVEY DATA INTO SQLITE
# =========================================================

def import_survey_data():
    if not os.path.exists(SURVEY_FILE):
        print("Survey data file not found:", SURVEY_FILE)
        return

    all_sheets = pd.read_excel(SURVEY_FILE, sheet_name=None)
    chosen_sheet_name, chosen_df = choose_best_survey_sheet(all_sheets)

    if chosen_df is None or chosen_df.empty:
        print("No usable survey sheet found.")
        return

    chosen_df = clean_survey_dataframe(chosen_df)

    conn = get_db_connection()
    chosen_df.to_sql("survey_data", conn, if_exists="replace", index=False)
    conn.close()

    print(f"Survey data imported into SQLite table: survey_data (sheet: {chosen_sheet_name})")


# =========================================================
# ML TRAINING
# =========================================================
# Train the ML model using the SQLite footfall_data table
# =========================================================

def train_model_from_db():
    conn = get_db_connection()

    try:
        df = pd.read_sql_query("SELECT * FROM footfall_data", conn)
    except Exception:
        conn.close()
        print("footfall_data table not available for ML training.")
        return None

    conn.close()

    if df.empty:
        print("footfall_data table is empty.")
        return None

    df.columns = df.columns.str.strip()

    required_columns = ["Total Visiting", "Total Passing through", "Total Footfall"]
    for col in required_columns:
        if col not in df.columns:
            print("Missing required column:", col)
            print("Available columns:", df.columns.tolist())
            return None

    df = df.dropna(subset=required_columns)

    df["Total Visiting"] = pd.to_numeric(df["Total Visiting"], errors="coerce")
    df["Total Passing through"] = pd.to_numeric(df["Total Passing through"], errors="coerce")
    df["Total Footfall"] = pd.to_numeric(df["Total Footfall"], errors="coerce")

    df = df.dropna(subset=required_columns)

    if df.empty:
        print("No valid rows left after cleaning.")
        return None

    X = df[["Total Visiting", "Total Passing through"]]
    y = df["Total Footfall"]

    model = LinearRegression()
    model.fit(X, y)

    return model


# =========================================================
# ML THRESHOLDS FOR FOOTFALL CATEGORY
# =========================================================

def calculate_footfall_thresholds():
    global FOOTFALL_THRESHOLDS

    conn = get_db_connection()

    try:
        df = pd.read_sql_query("SELECT * FROM footfall_data", conn)
    except Exception:
        conn.close()
        FOOTFALL_THRESHOLDS = {"low": 1000.0, "medium": 5000.0}
        return

    conn.close()

    if df.empty or "Total Footfall" not in df.columns:
        FOOTFALL_THRESHOLDS = {"low": 1000.0, "medium": 5000.0}
        return

    values = pd.to_numeric(df["Total Footfall"], errors="coerce").dropna()

    if values.empty:
        FOOTFALL_THRESHOLDS = {"low": 1000.0, "medium": 5000.0}
        return

    FOOTFALL_THRESHOLDS = {
        "low": float(values.quantile(0.33)),
        "medium": float(values.quantile(0.66))
    }


def get_footfall_level(value: Optional[float]) -> str:
    if value is None:
        return "Unknown"

    if value < FOOTFALL_THRESHOLDS["low"]:
        return "Low"
    elif value < FOOTFALL_THRESHOLDS["medium"]:
        return "Medium"
    return "High"


def predict_values(total_visiting, total_passing_through):
    if model is None:
        return None, "Unknown"

    try:
        tv = float(total_visiting)
        tp = float(total_passing_through)
    except (TypeError, ValueError):
        return None, "Unknown"

    input_df = pd.DataFrame([{
        "Total Visiting": tv,
        "Total Passing through": tp
    }])

    predicted_value = float(model.predict(input_df)[0])
    predicted_level = get_footfall_level(predicted_value)

    return round(predicted_value, 2), predicted_level


# =========================================================
# DATA CLEANING HELPERS
# =========================================================

def safe_float(value):
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def normalize_footfall_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()

    for col in [
        "Event Name",
        "Area",
        "area",
        "Date",
        "Total Footfall",
        "Total Visiting",
        "Total Passing through",
        "Event took place"
    ]:
        if col not in df.columns:
            df[col] = None

    df["Total Footfall"] = pd.to_numeric(df["Total Footfall"], errors="coerce")
    df["Total Visiting"] = pd.to_numeric(df["Total Visiting"], errors="coerce")
    df["Total Passing through"] = pd.to_numeric(df["Total Passing through"], errors="coerce")

    return df


def build_client_event_results(df: pd.DataFrame, limit: int = 50):
    if df.empty:
        return []

    df = normalize_footfall_dataframe(df)
    area_col = "Area" if "Area" in df.columns else "area"

    results = []
    for _, row in df.head(limit).iterrows():
        predicted_footfall, predicted_level = predict_values(
            row.get("Total Visiting"),
            row.get("Total Passing through")
        )

        results.append({
            "event_name": row.get("Event Name", ""),
            "area": row.get(area_col, ""),
            "date": row.get("Date", ""),
            "total_footfall": safe_float(row.get("Total Footfall")),
            "total_visiting": safe_float(row.get("Total Visiting")),
            "total_passing_through": safe_float(row.get("Total Passing through")),
            "event_took_place": row.get("Event took place", ""),
            "predicted_footfall": predicted_footfall,
            "predicted_level": predicted_level
        })

    return results


# =========================================================
# FASTAPI APP SETUP
# =========================================================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

init_db()
seed_users()
seed_events()
import_footfall_data()
import_survey_data()

model = train_model_from_db()
calculate_footfall_thresholds()

if model is not None:
    test_prediction, test_level = predict_values(50000, 100000)
    print("Predicted footfall:", test_prediction, "| Level:", test_level)
else:
    print("ML model was not trained.")


# =========================================================
# AUTH HELPERS
# =========================================================

def create_session(username: str, role: str) -> str:
    conn = get_db_connection()
    cursor = conn.cursor()

    token = secrets.token_hex(16)

    cursor.execute(
        "INSERT INTO sessions (token, username, role) VALUES (?, ?, ?)",
        (token, username, role)
    )
    conn.commit()
    conn.close()

    return token


def delete_session(token: str):
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("DELETE FROM sessions WHERE token = ?", (token,))
    conn.commit()
    conn.close()


def get_current_user(auth: str | None):
    if auth is None or not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing token")

    token = auth.split(" ", 1)[1]

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT username, role FROM sessions WHERE token = ?", (token,))
    session = cursor.fetchone()
    conn.close()

    if not session:
        raise HTTPException(status_code=401, detail="Invalid token")

    return {
        "token": token,
        "username": session["username"],
        "role": session["role"]
    }


def require_role(auth: str | None, allowed_roles: list[str]):
    user = get_current_user(auth)

    if user["role"] not in allowed_roles:
        raise HTTPException(status_code=403, detail="Not allowed")

    return user


# =========================================================
# REQUEST MODELS
# =========================================================

class LoginRequest(BaseModel):
    username: str
    password: str


class EventCreate(BaseModel):
    title: str
    date: str
    location: str
    footfall: int


class PredictRequest(BaseModel):
    total_visiting: float
    total_passing_through: float


class UserCreate(BaseModel):
    username: str
    password: str
    role: str


class UserRoleUpdate(BaseModel):
    role: str


class ClientEventSearchRequest(BaseModel):
    event_name: Optional[str] = None
    area: Optional[str] = None
    date: Optional[str] = None
    min_footfall: Optional[float] = None
    max_footfall: Optional[float] = None
    only_event_took_place: bool = False
    sort_by_predicted: bool = True


# =========================================================
# ROOT + FRONTEND
# =========================================================

@app.get("/")
def root():
    return {"message": "Event Management Dashboard backend running"}


@app.get("/dashboard")
def dashboard():
    return FileResponse("static/dashboard.html")


# =========================================================
# AUTH ROUTES
# =========================================================

@app.post("/login")
def login(req: LoginRequest):
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT * FROM users WHERE username = ?",
        (req.username,)
    )
    user = cursor.fetchone()
    conn.close()

    if not user or not verify_password(req.password, user["password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_session(user["username"], user["role"])

    return {
        "token": token,
        "role": user["role"],
        "username": user["username"]
    }


@app.post("/logout")
def logout(authorization: str | None = Header(default=None)):
    user = get_current_user(authorization)
    delete_session(user["token"])
    return {"status": "logged_out"}


@app.post("/seed-admin")
def seed_admin():
    return {"status": "ready", "username": "admin"}


# =========================================================
# USER MANAGEMENT (ADMIN ONLY)
# =========================================================

@app.get("/users")
def list_users(authorization: str | None = Header(default=None)):
    require_role(authorization, ["admin"])

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT id, username, role FROM users ORDER BY id ASC")
    rows = cursor.fetchall()
    conn.close()

    users = []
    for row in rows:
        users.append({
            "id": row["id"],
            "username": row["username"],
            "role": row["role"]
        })

    return users


@app.post("/users")
def create_user(user: UserCreate, authorization: str | None = Header(default=None)):
    require_role(authorization, ["admin"])

    if user.role not in ["admin", "staff", "viewer"]:
        raise HTTPException(status_code=400, detail="Invalid role")

    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute(
            "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
            (user.username, hash_password(user.password), user.role)
        )
        conn.commit()
        new_id = cursor.lastrowid
    except sqlite3.IntegrityError:
        conn.close()
        raise HTTPException(status_code=400, detail="Username already exists")

    conn.close()

    return {
        "status": "created",
        "user": {
            "id": new_id,
            "username": user.username,
            "role": user.role
        }
    }


@app.put("/users/{user_id}/role")
def update_user_role(user_id: int, data: UserRoleUpdate, authorization: str | None = Header(default=None)):
    current_user = require_role(authorization, ["admin"])

    if data.role not in ["admin", "staff", "viewer"]:
        raise HTTPException(status_code=400, detail="Invalid role")

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    target_user = cursor.fetchone()

    if not target_user:
        conn.close()
        raise HTTPException(status_code=404, detail="User not found")

    cursor.execute(
        "UPDATE users SET role = ? WHERE id = ?",
        (data.role, user_id)
    )
    conn.commit()

    cursor.execute(
        "UPDATE sessions SET role = ? WHERE username = ?",
        (data.role, target_user["username"])
    )
    conn.commit()
    conn.close()

    return {
        "status": "updated",
        "id": user_id,
        "new_role": data.role
    }


@app.delete("/users/{user_id}")
def delete_user(user_id: int, authorization: str | None = Header(default=None)):
    current_user = require_role(authorization, ["admin"])

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    target_user = cursor.fetchone()

    if not target_user:
        conn.close()
        raise HTTPException(status_code=404, detail="User not found")

    if target_user["username"] == current_user["username"]:
        conn.close()
        raise HTTPException(status_code=400, detail="Admin cannot delete own account while logged in")

    cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
    cursor.execute("DELETE FROM sessions WHERE username = ?", (target_user["username"],))
    conn.commit()
    conn.close()

    return {"status": "deleted", "id": user_id}


# =========================================================
# EVENTS (MANAGED APP EVENTS)
# =========================================================

@app.get("/events")
def list_events(authorization: str | None = Header(default=None)):
    require_role(authorization, ["admin", "staff", "viewer"])

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM events ORDER BY id ASC")
    rows = cursor.fetchall()
    conn.close()

    events = []
    for row in rows:
        events.append({
            "id": row["id"],
            "title": row["title"],
            "date": row["date"],
            "location": row["location"],
            "footfall": row["footfall"]
        })

    return events


@app.post("/events")
def create_event(event: EventCreate, authorization: str | None = Header(default=None)):
    require_role(authorization, ["admin", "staff"])

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO events (title, date, location, footfall) VALUES (?, ?, ?, ?)",
        (event.title, event.date, event.location, event.footfall)
    )
    conn.commit()

    new_id = cursor.lastrowid
    conn.close()

    return {
        "status": "created",
        "event": {
            "id": new_id,
            "title": event.title,
            "date": event.date,
            "location": event.location,
            "footfall": event.footfall
        }
    }


@app.delete("/events/{event_id}")
def delete_event(event_id: int, authorization: str | None = Header(default=None)):
    require_role(authorization, ["admin"])

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM events WHERE id = ?", (event_id,))
    event = cursor.fetchone()

    if not event:
        conn.close()
        raise HTTPException(status_code=404, detail="Event not found")

    cursor.execute("DELETE FROM events WHERE id = ?", (event_id,))
    conn.commit()
    conn.close()

    return {"status": "deleted", "id": event_id}


# =========================================================
# ML PREDICTION
# =========================================================

@app.post("/predict")
def predict_footfall(data: PredictRequest, authorization: str | None = Header(default=None)):
    require_role(authorization, ["admin", "staff"])

    if model is None:
        raise HTTPException(status_code=500, detail="ML model not available")

    predicted_value, predicted_level = predict_values(
        data.total_visiting,
        data.total_passing_through
    )

    return {
        "predicted_footfall": predicted_value,
        "predicted_level": predicted_level
    }


# =========================================================
# READ REAL CLIENT DATA
# =========================================================

@app.get("/footfall-data")
def view_footfall_data(authorization: str | None = Header(default=None)):
    require_role(authorization, ["admin"])

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM footfall_data LIMIT 20")
    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]


@app.get("/survey-data")
def view_survey_data(authorization: str | None = Header(default=None)):
    require_role(authorization, ["admin"])

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM survey_data LIMIT 20")
    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]


# =========================================================
# SURVEY ANALYTICS
# =========================================================

@app.get("/survey-summary")
def survey_summary(authorization: str | None = Header(default=None)):
    require_role(authorization, ["admin"])

    conn = get_db_connection()

    try:
        df = pd.read_sql_query("SELECT * FROM survey_data", conn)
    except Exception:
        conn.close()
        raise HTTPException(status_code=500, detail="Survey data not available")

    conn.close()

    if df.empty:
        return {
            "total_responses": 0,
            "feedback_categories": {},
            "suggestions": {}
        }

    df = clean_survey_dataframe(df)

    # Find all category-like columns
    category_like_columns = [col for col in df.columns if "category" in str(col).lower()]

    # Build counts for each category-like column
    column_counts = []
    for col in category_like_columns:
        counts = count_meaningful_values(df[col])
        if counts:
            column_counts.append((col, counts))

    if not column_counts:
        return {
            "total_responses": len(df),
            "feedback_categories": {},
            "suggestions": {}
        }

    # Use the first meaningful category column as feedback
    feedback_counts = column_counts[0][1]

    # Merge all the other meaningful category columns as suggestions
    suggestion_counts = {}
    for _, counts in column_counts[1:]:
        for key, value in counts.items():
            suggestion_counts[key] = suggestion_counts.get(key, 0) + value

    return {
        "total_responses": len(df),
        "feedback_categories": feedback_counts,
        "suggestions": suggestion_counts
    }


# =========================================================
# REAL CLIENT EVENT DATA
# =========================================================

@app.get("/client-events")
def client_events(authorization: str | None = Header(default=None)):
    require_role(authorization, ["admin", "staff", "viewer"])

    conn = get_db_connection()

    try:
        df = pd.read_sql_query("SELECT * FROM footfall_data", conn)
    except Exception:
        conn.close()
        raise HTTPException(status_code=500, detail="Footfall data not available")

    conn.close()

    if df.empty:
        return []

    df = normalize_footfall_dataframe(df)

    event_name_series = df["Event Name"].fillna("").astype(str).str.strip()
    event_took_place_series = df["Event took place"].fillna("").astype(str).str.strip().str.lower()

    event_like = df[
        (event_name_series != "") |
        (~event_took_place_series.isin(["", "0", "no", "false", "nan"]))
    ].copy()

    if event_like.empty:
        event_like = df.sort_values("Total Footfall", ascending=False).head(20).copy()

    return build_client_event_results(event_like, limit=20)


# =========================================================
# SMART CLIENT EVENT SEARCH
# =========================================================

@app.post("/client-events/search")
def search_client_events(
    filters: ClientEventSearchRequest,
    authorization: str | None = Header(default=None)
):
    require_role(authorization, ["admin", "staff", "viewer"])

    conn = get_db_connection()

    try:
        df = pd.read_sql_query("SELECT * FROM footfall_data", conn)
    except Exception:
        conn.close()
        raise HTTPException(status_code=500, detail="Footfall data not available")

    conn.close()

    if df.empty:
        return {
            "count": 0,
            "results": []
        }

    df = normalize_footfall_dataframe(df)
    area_col = "Area" if "Area" in df.columns else "area"

    if filters.event_name:
        search_value = filters.event_name.strip().lower()
        df = df[
            df["Event Name"].fillna("").astype(str).str.lower().str.contains(search_value, na=False)
        ]

    if filters.area:
        search_area = filters.area.strip().lower()
        df = df[
            df[area_col].fillna("").astype(str).str.lower().str.contains(search_area, na=False)
        ]

    if filters.date:
        search_date = filters.date.strip().lower()
        df = df[
            df["Date"].fillna("").astype(str).str.lower().str.contains(search_date, na=False)
        ]

    if filters.min_footfall is not None:
        df = df[df["Total Footfall"] >= filters.min_footfall]

    if filters.max_footfall is not None:
        df = df[df["Total Footfall"] <= filters.max_footfall]

    if filters.only_event_took_place:
        event_took_place_series = df["Event took place"].fillna("").astype(str).str.strip().str.lower()
        df = df[
            ~event_took_place_series.isin(["", "0", "no", "false", "nan"])
        ]

    if df.empty:
        return {
            "count": 0,
            "results": []
        }

    results = build_client_event_results(df, limit=50)

    if filters.sort_by_predicted:
        results.sort(
            key=lambda item: item["predicted_footfall"] if item["predicted_footfall"] is not None else -1,
            reverse=True
        )

    return {
        "count": len(results),
        "results": results
    }