import io
import base64
import re
from datetime import datetime, timedelta, time
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import locale

locale.setlocale(locale.LC_TIME, '')


def generate_report_html(data):
    messages = data['messages']

    # === 1. Разбор сообщений ===
    sleep_pattern = re.compile(r"(\d{2}:\d{2})-(\d{2}:\d{2})")
    feed_pattern = re.compile(r"(\d{2}:\d{2})\s*([Лл]ев|[Пп]рав)")

    sleep_records = []
    feed_records = []

    for msg in messages:
        text = msg.get("text", "")
        if isinstance(text, list):
            text = " ".join(t if isinstance(t, str) else t.get("text", "") for t in text)
        if not text:
            continue

        dt = datetime.fromtimestamp(msg['date'])

        sleep_match = sleep_pattern.search(text)
        if sleep_match:
            start_str, end_str = sleep_match.groups()
            start_time = datetime.strptime(start_str, "%H:%M").time()
            end_time = datetime.strptime(end_str, "%H:%M").time()
            candidate_start = datetime.combine(dt.date(), start_time)
            candidate_end = datetime.combine(dt.date(), end_time)
            if candidate_end < candidate_start:
                candidate_end += timedelta(days=1)
            sleep_records.append((candidate_start, candidate_end))
            continue

        feed_match = feed_pattern.search(text)
        if feed_match:
            feed_time_str, side = feed_match.groups()
            feed_time = datetime.strptime(feed_time_str, "%H:%M").time()
            feed_dt = datetime.combine(dt.date(), feed_time)
            feed_records.append((feed_dt, side))

    sleep_df = pd.DataFrame(sleep_records, columns=["start", "end"])
    sleep_df = sleep_df.sort_values("start")
    feed_df = pd.DataFrame(feed_records, columns=["time", "side"])
    feed_df = feed_df.sort_values("time")

    # === 2. Расчёт длительности снов ===
    sleep_df["duration"] = (sleep_df["end"] - sleep_df["start"]).dt.total_seconds() / 60

    # === 3. График временной линии сна ===
    plt.figure(figsize=(10, 2))
    for idx, row in sleep_df.iterrows():
        plt.plot([row["start"], row["end"]], [idx, idx], color="blue", linewidth=5)
    plt.title("Временная линия сна")
    plt.yticks([])
    plt.tight_layout()
    buf_timeline = io.BytesIO()
    plt.savefig(buf_timeline, format="png")
    buf_timeline.seek(0)
    timeline_b64 = base64.b64encode(buf_timeline.read()).decode("utf-8")
    plt.close()

    # === 4. Гистограмма длительностей сна ===
    plt.figure(figsize=(6, 3))
    plt.hist(sleep_df["duration"], bins=20, color="skyblue", edgecolor="black")
    plt.title("Распределение длительностей сна (мин)")
    plt.xlabel("минуты")
    plt.ylabel("частота")
    plt.tight_layout()
    buf_hist = io.BytesIO()
    plt.savefig(buf_hist, format="png")
    buf_hist.seek(0)
    hist_b64 = base64.b64encode(buf_hist.read()).decode("utf-8")
    plt.close()

    # === 5. Расчёт времени бодрствования ===
    awake_durations = []
    for i in range(1, len(sleep_df)):
        prev_end = sleep_df.iloc[i - 1]["end"]
        current_start = sleep_df.iloc[i]["start"]
        diff = (current_start - prev_end).total_seconds() / 60
        if 0 < diff < 1000:  # фильтр странных разрывов
            awake_durations.append(diff)
    avg_awake = np.mean(awake_durations) if awake_durations else 0

    # === 6. HTML отчет ===
    html = f"""
    <html>
    <head>
        <meta charset='utf-8'>
        <title>Baby Tracker Report</title>
        <style>
            body {{ font-family: sans-serif; max-width: 900px; margin: 2rem auto; }}
            img {{ max-width: 100%; display: block; margin-bottom: 2rem; }}
            h1 {{ margin-bottom: 0.5rem; }}
        </style>
    </head>
    <body>
        <h1>Baby Tracker Report</h1>
        <p>Сгенерировано: {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>
        <p>Среднее время бодрствования: {avg_awake:.0f} мин.</p>
        <h2>Временная линия сна</h2>
        <img src="data:image/png;base64,{timeline_b64}">
        <h2>Распределение длительностей сна</h2>
        <img src="data:image/png;base64,{hist_b64}">
    </body>
    </html>
    """
    return html
