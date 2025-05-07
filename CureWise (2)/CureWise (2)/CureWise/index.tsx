// -- Create the database
// CREATE DATABASE IF NOT EXISTS curewise;
//
// -- Use the database
// USE curewise;
//
// -- Create the users table
// CREATE TABLE IF NOT EXISTS users (
//     id INT AUTO_INCREMENT PRIMARY KEY,
//     name VARCHAR(255) NOT NULL,
//     email VARCHAR(255) NOT NULL UNIQUE,
//     password VARCHAR(255) NOT NULL,
//     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
// );
//
// -- Create the medications table
// CREATE TABLE IF NOT EXISTS medications (
//     id INT AUTO_INCREMENT PRIMARY KEY,
//     user_id INT,
//     medication_name VARCHAR(255) NOT NULL,
//     dosage VARCHAR(100),
//     frequency VARCHAR(100),
//     start_date DATE,
//     end_date DATE,
//     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
//     FOREIGN KEY (user_id) REFERENCES users(id)
// );
//
// -- Create the health_records table
// CREATE TABLE IF NOT EXISTS health_records (
//     id INT AUTO_INCREMENT PRIMARY KEY,
//     user_id INT,
//     record_type VARCHAR(100) NOT NULL,
//     record_value TEXT,
//     record_date DATE,
//     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
//     FOREIGN KEY (user_id) REFERENCES users(id)
// );