from flask import Flask, send_file
import subprocess
import os

app = Flask(__name__)

@app.route('/')
def index():
    # Запускаем  скрипт
    subprocess.run(["python", "new_awaketime.py"], check=True)
    # Возвращаем HTML-отчет
    return send_file("last3days_report.html")

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
