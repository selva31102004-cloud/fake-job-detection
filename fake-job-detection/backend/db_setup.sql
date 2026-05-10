-- ─────────────────────────────────────────────────────────────────────────────
-- FraudShield — MySQL Database Setup
-- Run this once before starting the backend.
-- ─────────────────────────────────────────────────────────────────────────────

-- 1. Create the database (if it doesn't already exist)
CREATE DATABASE IF NOT EXISTS login_system
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

USE login_system;

-- 2. Create a dedicated app user (replace 'strongpassword' with a real one)
-- CREATE USER IF NOT EXISTS 'fraudshield_app'@'localhost' IDENTIFIED BY 'strongpassword';
-- GRANT SELECT, INSERT, UPDATE, DELETE ON fraudshield.* TO 'fraudshield_app'@'localhost';
-- FLUSH PRIVILEGES;

-- 3. Users table (auto-created by the API on startup, but included here for reference)
CREATE TABLE IF NOT EXISTS users (
    id            INT AUTO_INCREMENT PRIMARY KEY,
    first_name    VARCHAR(100)         NOT NULL,
    last_name     VARCHAR(100)         NOT NULL,
    email         VARCHAR(255) UNIQUE  NOT NULL,
    password_hash VARCHAR(128)         NOT NULL,
    salt          VARCHAR(64)          NOT NULL,
    is_active     TINYINT(1)           NOT NULL DEFAULT 1,
    last_login    DATETIME             NULL,
    created_at    DATETIME             NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at    DATETIME             NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_email (email)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ─── Verify ───────────────────────────────────────────────────────────────────
SHOW TABLES;
DESCRIBE users;
