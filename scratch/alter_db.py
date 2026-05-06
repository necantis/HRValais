import sqlite3

def alter():
    conn = sqlite3.connect('hr_valais_prototype.db')
    cursor = conn.cursor()
    try:
        cursor.execute("ALTER TABLE users ADD COLUMN max_surveys_per_year INTEGER DEFAULT 1;")
        print("Column added.")
    except Exception as e:
        print("Error adding column (maybe already exists):", e)
        
    try:
        cursor.execute("DELETE FROM users;")
        cursor.execute("DELETE FROM firms;")
        cursor.execute("DELETE FROM survey_responses;")
        print("Cleared data.")
    except Exception as e:
        print("Error clearing:", e)

    conn.commit()
    conn.close()

if __name__ == "__main__":
    alter()
