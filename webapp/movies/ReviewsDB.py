import os
import sqlite3


def db_setup(name):
    dest = os.path.join(name + '.sqllite')
    # if os.path.exists(dest):
    #     os.remove(dest)
    conn = sqlite3.connect(dest)
    cursor = conn.cursor()
    # cursor.execute('CREATE TABLE review_db (review TEXT, sentiment INTEGER, date TEXT)')

    return cursor, conn


if __name__ == "__main__":
    _cursor, _conn = db_setup('reviews')

    example1 = 'I love this movie'
    _cursor.execute("INSERT INTO review_db" \
                    " (review, sentiment, date) VALUES" \
                    " (?, ?, DATETIME('now'))", (example1, 1))

    _conn.commit()

    _cursor.execute("SELECT * FROM review_db WHERE date" \
                    " BETWEEN '2017-01-01 00:00:00' AND DATETIME('now')")
    results = _cursor.fetchall()
    _conn.close()

    print(results)
