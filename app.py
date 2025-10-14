import os
from flask import Flask, render_template_string
import json
import boto3
from awaken_time import generate_report_html

# можно безопасно хранить ключи в переменных окружения
ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID", "ТВОЙ_ACCESS_KEY_ID")
SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "ТВОЙ_SECRET_ACCESS_KEY")
ENDPOINT = "https://2f788f84f16189ad731022390c8d0b13.r2.cloudflarestorage.com"
BUCKET = "baby-tracker-data"
KEY = "result.json"

s3 = boto3.client(
    "s3",
    endpoint_url=ENDPOINT,
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
    region_name="auto"
)

app = Flask(__name__)

@app.route("/")
def index():
    # Получаем свежие данные
    response = s3.get_object(Bucket=BUCKET, Key=KEY)
    data = json.loads(response["Body"].read().decode("utf-8"))

    # Генерация отчёта
    html = generate_report_html(data)
    return render_template_string(html)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
