import sqlite3
import pyotp
import bcrypt
import jwt
import datetime
import os
from typing import Optional, Dict

DB_PATH = "users.db"
JWT_SECRET = os.getenv("JWT_SECRET", "super-secret-key-change-in-prod")
ALGORITHM = "HS256"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            totp_secret TEXT,
            is_2fa_enabled BOOLEAN DEFAULT 0,
            xm_account TEXT,
            xm_password TEXT,
            xm_server TEXT
        )
    """)
    conn.commit()
    conn.close()

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def hash_password(password: str) -> str:
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))

def create_user(email: str, password: str) -> Dict:
    conn = get_db()
    cursor = conn.cursor()
    
    # Check if user exists
    cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
    if cursor.fetchone():
        conn.close()
        raise ValueError("User already exists")

    pwd_hash = hash_password(password)
    totp_secret = pyotp.random_base32()
    
    cursor.execute(
        "INSERT INTO users (email, password_hash, totp_secret, is_2fa_enabled) VALUES (?, ?, ?, 0)",
        (email, pwd_hash, totp_secret)
    )
    user_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    # Generate provisioning URI for QR code
    totp = pyotp.TOTP(totp_secret)
    uri = totp.provisioning_uri(name=email, issuer_name="Stock Predictor Agent")
    return {"user_id": user_id, "totp_secret": totp_secret, "qr_uri": uri}

def verify_totp(email: str, code: str, enable: bool = False) -> bool:
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT totp_secret FROM users WHERE email = ?", (email,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        return False
        
    totp = pyotp.TOTP(row["totp_secret"])
    is_valid = totp.verify(code, valid_window=20)
    
    if is_valid and enable:
        cursor.execute("UPDATE users SET is_2fa_enabled = 1 WHERE email = ?", (email,))
        conn.commit()
        
    conn.close()
    return is_valid

def authenticate_user(email: str, password: str, code: str) -> Optional[Dict]:
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
    user = cursor.fetchone()
    conn.close()
    
    if not user:
        return None
    if not verify_password(password, user["password_hash"]):
        return None
    if user["is_2fa_enabled"] and not verify_totp(email, code):
        return None
        
    return dict(user)

def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.datetime.utcnow() + datetime.timedelta(hours=24)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, JWT_SECRET, algorithm=ALGORITHM)

def decode_token(token: str) -> Optional[dict]:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[ALGORITHM])
    except:
        return None

def update_xm_credentials(email: str, xm_account: str, xm_password: str, xm_server: str):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE users SET xm_account = ?, xm_password = ?, xm_server = ? WHERE email = ?",
        (xm_account, xm_password, xm_server, email)
    )
    conn.commit()
    conn.close()

# Initialize DB on import
init_db()
