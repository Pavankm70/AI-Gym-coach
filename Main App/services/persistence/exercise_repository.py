import sqlite3
from pathlib import Path

try:
    import streamlit as st
    cache_resource = st.cache_resource
except Exception:
    def cache_resource(f):
        return f

_DB_PATH = str(Path(__file__).parent.parent.parent / "data.db")
_global_conn = None


@cache_resource
def _get_connection() -> sqlite3.Connection:
    global _global_conn
    if _global_conn is None:
        _global_conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
        _global_conn.row_factory = sqlite3.Row
    return _global_conn


def init_db() -> None:
    conn = _get_connection()

    with conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                username   TEXT UNIQUE NOT NULL,
                fitness_level TEXT DEFAULT 'Beginner',
                goals TEXT DEFAULT 'General Fitness',
                preferred_feedback_style TEXT DEFAULT 'Encouraging',
                experience TEXT DEFAULT 'Beginner (0-1 years)',
                workout_frequency TEXT DEFAULT '3 days/week',
                limitations TEXT DEFAULT 'None',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        # Ensure new columns exist for existing databases
        columns_to_add = [
            ("fitness_level", "TEXT DEFAULT 'Beginner'"),
            ("goals", "TEXT DEFAULT 'General Fitness'"),
            ("preferred_feedback_style", "TEXT DEFAULT 'Encouraging'"),
            ("experience", "TEXT DEFAULT 'Beginner (0-1 years)'"),
            ("workout_frequency", "TEXT DEFAULT '3 days/week'"),
            ("limitations", "TEXT DEFAULT 'None'"),
            ("coaching_intensity", "TEXT DEFAULT 'ERRORS_ONLY'"),
        ]
        existing_cols = [row[1] for row in conn.execute("PRAGMA table_info(users)").fetchall()]
        for col_name, col_def in columns_to_add:
            if col_name not in existing_cols:
                try:
                    conn.execute(f"ALTER TABLE users ADD COLUMN {col_name} {col_def}")
                except Exception:
                    pass

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS exercises (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id       INTEGER NOT NULL REFERENCES users(id),
                exercise_name TEXT    NOT NULL,
                reps          INTEGER NOT NULL DEFAULT 0,
                sets          INTEGER NOT NULL DEFAULT 0,
                time          INTEGER NOT NULL DEFAULT 0,
                created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )


def get_user(username: str) -> sqlite3.Row:
    conn = _get_connection()

    return conn.execute(
        "SELECT * FROM users WHERE username = ?", (username,)
    ).fetchone()


def get_user_by_id(user_id: int) -> sqlite3.Row:
    conn = _get_connection()
    return conn.execute(
        "SELECT * FROM users WHERE id = ?", (user_id,)
    ).fetchone()


def update_user_profile(user_id: int, profile_data: dict) -> sqlite3.Row:
    conn = _get_connection()
    allowed_fields = [
        "fitness_level",
        "goals",
        "preferred_feedback_style",
        "experience",
        "workout_frequency",
        "limitations"
    ]
    updates = []
    values = []
    for k in allowed_fields:
        if k in profile_data:
            updates.append(f"{k} = ?")
            values.append(profile_data[k])
    if updates:
        values.append(user_id)
        with conn:
            conn.execute(
                f"UPDATE users SET {', '.join(updates)} WHERE id = ?",
                tuple(values)
            )
    return get_user_by_id(user_id)


def create_user(username: str) -> sqlite3.Row:
    conn = _get_connection()
    
    with conn:
        conn.execute(
            "INSERT INTO users (username) VALUES (?)", (username,)
        )

    return get_user(username) 


def get_or_create_user(username: str) -> sqlite3.Row:
    user = get_user(username)

    if user is None:
        user = create_user(username)
    
    return user


def add_exercise(user_id, exercise_name, reps, sets, time):
    conn = _get_connection()

    with conn:
        existing = conn.execute("""
            SELECT * FROM exercises 
            WHERE user_id = ? AND exercise_name = ? AND Date('created_at') = Date('now')
        """, (user_id, exercise_name)).fetchone()

        if existing:
            conn.execute("""
                UPDATE exercises 
                SET reps = reps + ?, sets = sets + ?, time = time + ?
                WHERE id = ?
            """, (reps, sets, time, existing['id']))
        else:
            conn.execute("""
                INSERT INTO exercises (user_id, exercise_name, sets, reps, time)
                VALUES (?, ?, ?, ?, ?)
            """, (user_id, exercise_name, sets, reps, time))


def get_users_exercises(user_id):
    conn = _get_connection()

    return conn.execute("""
        SELECT * FROM exercises 
        WHERE user_id = ?
    """, (user_id,)).fetchall()
