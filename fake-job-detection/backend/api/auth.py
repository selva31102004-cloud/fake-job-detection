"""
User Authentication Module — MySQL Backend
FraudShield Fake Job Detection System
"""

import os
import hashlib
import secrets
import time
from datetime import datetime, timedelta
from typing import Optional

import mysql.connector
from mysql.connector import Error as MySQLError
from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
import jwt

# ─── CONFIG (override via environment variables) ──────────────────────────────
DB_HOST     = os.getenv("DB_HOST", "localhost")
DB_PORT     = int(os.getenv("DB_PORT", "3306"))
DB_NAME     = os.getenv("DB_NAME", "login_system")
DB_USER     = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "Selv@R@1")
JWT_SECRET  = os.getenv("JWT_SECRET", "change-me-in-production-use-a-long-random-string")
JWT_ALGO    = "HS256"
TOKEN_EXPIRE_HOURS = 24
REMEMBER_EXPIRE_DAYS = 30

# ─── SCHEMAS ──────────────────────────────────────────────────────────────────
class RegisterRequest(BaseModel):
    first_name: str
    last_name: str
    email: EmailStr
    password: str

class LoginRequest(BaseModel):
    email: EmailStr
    password: str
    remember_me: bool = False

class UserOut(BaseModel):
    id: int
    first_name: str
    last_name: str
    email: str
    created_at: str

class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut

# ─── DATABASE ─────────────────────────────────────────────────────────────────
def get_connection():
    """Return a fresh MySQL connection."""
    return mysql.connector.connect(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        charset="utf8mb4",
        use_unicode=True,
        autocommit=False,
    )

def init_db():
    """
    Create the users table if it doesn't exist.
    Call this once at startup.
    """
    ddl = """
    CREATE TABLE IF NOT EXISTS users (
        id           INT AUTO_INCREMENT PRIMARY KEY,
        first_name   VARCHAR(100)        NOT NULL,
        last_name    VARCHAR(100)        NOT NULL,
        email        VARCHAR(255) UNIQUE NOT NULL,
        password_hash VARCHAR(128)       NOT NULL,
        salt         VARCHAR(64)        NOT NULL,
        is_active    TINYINT(1)         NOT NULL DEFAULT 1,
        last_login   DATETIME           NULL,
        created_at   DATETIME           NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at   DATETIME           NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        INDEX idx_email (email)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
    """
    try:
        conn = get_connection()
        cur  = conn.cursor()
        cur.execute(ddl)
        conn.commit()
        cur.close()
        conn.close()
    except MySQLError as e:
        raise RuntimeError(f"DB init failed: {e}")

# ─── PASSWORD HELPERS ─────────────────────────────────────────────────────────
def _hash_password(password: str, salt: str) -> str:
    """SHA-256 with salt (swap for bcrypt/argon2 in production)."""
    return hashlib.sha256(f"{salt}{password}".encode()).hexdigest()

def create_password_hash(password: str) -> tuple[str, str]:
    """Return (hash, salt)."""
    salt = secrets.token_hex(32)
    return _hash_password(password, salt), salt

def verify_password(password: str, stored_hash: str, salt: str) -> bool:
    return secrets.compare_digest(_hash_password(password, salt), stored_hash)

# ─── JWT HELPERS ──────────────────────────────────────────────────────────────
def create_token(user_id: int, remember: bool = False) -> str:
    expire = timedelta(days=REMEMBER_EXPIRE_DAYS) if remember else timedelta(hours=TOKEN_EXPIRE_HOURS)
    payload = {
        "sub": str(user_id),
        "iat": datetime.utcnow(),
        "exp": datetime.utcnow() + expire,
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGO)

def decode_token(token: str) -> int:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGO])
        return int(payload["sub"])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired. Please sign in again.")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token.")

# ─── DEPENDENCY: current user ─────────────────────────────────────────────────
security = HTTPBearer()

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    user_id = decode_token(credentials.credentials)
    try:
        conn = get_connection()
        cur  = conn.cursor(dictionary=True)
        cur.execute("SELECT id, first_name, last_name, email, created_at FROM users WHERE id=%s AND is_active=1", (user_id,))
        user = cur.fetchone()
        cur.close(); conn.close()
    except MySQLError as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")

    if not user:
        raise HTTPException(status_code=401, detail="User not found.")
    return user

# ─── ROUTER ───────────────────────────────────────────────────────────────────
router = APIRouter(prefix="/auth", tags=["auth"])

@router.post("/register", response_model=AuthResponse, status_code=201)
def register(req: RegisterRequest):
    """Create a new user account and return a JWT."""
    if len(req.password) < 8:
        raise HTTPException(status_code=422, detail="Password must be at least 8 characters.")

    password_hash, salt = create_password_hash(req.password)

    try:
        conn = get_connection()
        cur  = conn.cursor(dictionary=True)

        # Check duplicate email
        cur.execute("SELECT id FROM users WHERE email = %s", (req.email,))
        if cur.fetchone():
            raise HTTPException(status_code=409, detail="An account with that email already exists.")

        # Insert new user
        cur.execute(
            """INSERT INTO users (first_name, last_name, email, password_hash, salt)
               VALUES (%s, %s, %s, %s, %s)""",
            (req.first_name.strip(), req.last_name.strip(), req.email, password_hash, salt)
        )
        conn.commit()
        user_id = cur.lastrowid

        # Fetch back for response
        cur.execute("SELECT id, first_name, last_name, email, created_at FROM users WHERE id=%s", (user_id,))
        user = cur.fetchone()
        cur.close(); conn.close()

    except HTTPException:
        raise
    except MySQLError as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")

    token = create_token(user_id)
    return AuthResponse(
        access_token=token,
        user=UserOut(
            id=user["id"],
            first_name=user["first_name"],
            last_name=user["last_name"],
            email=user["email"],
            created_at=str(user["created_at"]),
        )
    )


@router.post("/login", response_model=AuthResponse)
def login(req: LoginRequest):
    """Authenticate and return a JWT."""
    try:
        conn = get_connection()
        cur  = conn.cursor(dictionary=True)
        cur.execute(
            "SELECT id, first_name, last_name, email, password_hash, salt, is_active, created_at FROM users WHERE email=%s",
            (req.email,)
        )
        user = cur.fetchone()
    except MySQLError as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")

    if not user or not verify_password(req.password, user["password_hash"], user["salt"]):
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    if not user["is_active"]:
        raise HTTPException(status_code=403, detail="Account is disabled. Please contact support.")

    # Update last_login
    try:
        cur.execute("UPDATE users SET last_login=NOW() WHERE id=%s", (user["id"],))
        conn.commit()
        cur.close(); conn.close()
    except MySQLError:
        pass  # non-fatal

    token = create_token(user["id"], remember=req.remember_me)
    return AuthResponse(
        access_token=token,
        user=UserOut(
            id=user["id"],
            first_name=user["first_name"],
            last_name=user["last_name"],
            email=user["email"],
            created_at=str(user["created_at"]),
        )
    )


@router.get("/me", response_model=UserOut)
def me(current_user: dict = Depends(get_current_user)):
    """Return the currently authenticated user's profile."""
    return UserOut(
        id=current_user["id"],
        first_name=current_user["first_name"],
        last_name=current_user["last_name"],
        email=current_user["email"],
        created_at=str(current_user["created_at"]),
    )


@router.post("/logout")
def logout():
    """
    Stateless JWT: client simply deletes the token.
    For server-side revocation, implement a token blocklist in Redis/MySQL.
    """
    return {"message": "Logged out successfully. Delete your local token."}
