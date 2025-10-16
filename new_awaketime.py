import json
import re
from datetime import datetime, timedelta, time
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # вот эта строка спасает от ошибки с Tcl/Tk
import matplotlib.pyplot as plt
import os
TOKEN = os.getenv("TELEGRAM_TOKEN")
import numpy as np
import locale

import boto3
import io, base64

buf = io.BytesIO()
plt.savefig(buf, format="png")
plt.close()
buf.seek(0)
img_b64 = base64.b64encode(buf.read()).decode("utf-8")
html += f'<img src="data:image/png;base64,{img_b64}">'



# Настройка клиента R2
s3 = boto3.client(
    's3',
    endpoint_url='https://2f788f84f16189ad731022390c8d0b13.r2.cloudflarestorage.com',
    aws_access_key_id = os.getenv('AWS_ACCESS_KEY'),
    aws_secret_access_key = os.getenv('AWS_SECRET_KEY'),

region_name='auto'
)

# Считываем файл из R2 прямо в память
response = s3.get_object(Bucket='baby-tracker-data', Key='result.json')
data = json.loads(response['Body'].read().decode('utf-8'))

messages = data['messages']

# Скачивание файла
#s3.download_file('baby-tracker-data', 'result.json')

# Или чтение напрямую в память
#response = s3.get_object(Bucket='baby-tracker-data', Key='result.json')
#data = response['Body'].read()

# === 1. Настройки ===
#INPUT_FILE = 'result.json'

# === 2. Загрузка данных ===
#with open(INPUT_FILE, 'r', encoding='utf-8') as f:
#    data = json.load(f)

messages = data['messages']

sleep_data = []
feed_data = []
bath_data = []  # события ванна
toilet_data = []  # события покакал

# === 3. Регулярные выражения ===
sleep_pattern = re.compile(r'(\d{1,2}:\d{2})[–\-](\d{1,2}:\d{2}).*сон', re.IGNORECASE)
feed_pattern = re.compile(r'(\d{1,2}:\d{2}) смесь[^\d]*(\d+)', re.IGNORECASE)
bath_pattern = re.compile(r'(\d{1,2}:\d{2}).*ванна', re.IGNORECASE)
toilet_pattern = re.compile(r'(\d{1,2}:\d{2}).*покакал', re.IGNORECASE)

# === 4. Парсинг сообщений ===
for msg in messages:
    if msg.get('type') != 'message':
        continue

    text = msg.get('text')
    if isinstance(text, list):
        text = ''.join([t if isinstance(t, str) else t.get('text', '') for t in text])

    msg_date = datetime.fromisoformat(msg['date'])

    # Сон
    sleep_match = sleep_pattern.search(text)
    if sleep_match:
        start_str, end_str = sleep_match.groups()
        start_time = datetime.strptime(start_str, "%H:%M").time()
        end_time = datetime.strptime(end_str, "%H:%M").time()

        # кандидаты на ту же дату сообщения
        candidate_start = datetime.combine(msg_date.date(), start_time)
        candidate_end = datetime.combine(msg_date.date(), end_time)
        # если конец не позже старта — пересекаем полночь
        if end_time <= start_time:
            candidate_end += timedelta(days=1)

        msg_date = msg_date.replace(tzinfo=None)
        candidate_start = candidate_start.replace(tzinfo=None)

        # Новый блок сдвига: переносим на предыдущий день ТОЛЬКО если:
        # - сообщение пришло до 5 утра
        # - и начало сна позднее 18:00 (вечер предыдущего дня)
        if (
                msg_date.time() < time(5, 0)
                and candidate_start.time() > time(18, 0)
                and candidate_start > msg_date
        ):
            start_dt = candidate_start - timedelta(days=1)
            end_dt = candidate_end - timedelta(days=1)
        else:
            start_dt = candidate_start
            end_dt = candidate_end

        # DEBUG-строка может помочь при отладке
        # print("DEBUG:", text, start_dt, end_dt)
        sleep_data.append({'date': start_dt.date(), 'start': start_dt, 'end': end_dt})
        continue

    # Смесь
    feed_match = feed_pattern.search(text)
    if feed_match:
        feed_time = datetime.strptime(feed_match.group(1), "%H:%M").time()
        amount = int(feed_match.group(2))
        feed_dt = datetime.combine(msg_date.date(), feed_time)
        feed_data.append({'date': msg_date.date(), 'time': feed_dt, 'amount': amount})
        continue

    # Ванна
    bath_match = bath_pattern.search(text)
    if bath_match:
        bath_time = datetime.strptime(bath_match.group(1), "%H:%M").time()
        bath_dt = datetime.combine(msg_date.date(), bath_time)
        bath_data.append({'date': msg_date.date(), 'time': bath_dt})
        continue

    # Покакал
    toilet_match = toilet_pattern.search(text)
    if toilet_match:
        toilet_time = datetime.strptime(toilet_match.group(1), "%H:%M").time()
        toilet_dt = datetime.combine(msg_date.date(), toilet_time)
        toilet_data.append({'date': msg_date.date(), 'time': toilet_dt})
        continue

# === 5. В датафреймы ===
sleep_df = pd.DataFrame(sleep_data)
sleep_df = sleep_df.drop_duplicates(subset=['start', 'end'])
feed_df = pd.DataFrame(feed_data)
bath_df = pd.DataFrame(bath_data)
toilet_df = pd.DataFrame(toilet_data)

# === 6. Построение графика (лента событий) ===
# Собираем все даты (универсально: берем диапазон от минимальной до максимальной даты событий)
dates_from = []
if not feed_df.empty:
    dates_from.extend(list(feed_df['date'].unique()))
if not sleep_df.empty:
    # use both start and end dates to cover sleeps, которые переходят через полночь
    dates_from.extend(list(sleep_df['start'].dt.date.unique()))
    dates_from.extend(list(sleep_df['end'].dt.date.unique()))

if dates_from:
    min_date = min(dates_from)
    max_date = max(dates_from)
    all_dates = [d.date() if isinstance(d, datetime) else d for d in pd.date_range(min_date, max_date).date]
else:
    all_dates = []

fig1, ax = plt.subplots(figsize=(12, max(1, len(all_dates)) * 0.6))

cmap = plt.cm.Blues

def get_color_by_amount(amount):
    if amount <= 40:
        return cmap(0.3)
    elif amount <= 70:
        return cmap(0.5)
    elif amount <= 100:
        return cmap(0.7)
    else:
        return cmap(0.9)

for i, day in enumerate(all_dates):
    base_time = datetime.combine(day, datetime.min.time())

    # Сон
    if not sleep_df.empty:
        for _, row in sleep_df.iterrows():
            start = row['start']
            end = row['end']
            current_day_start = datetime.combine(day, datetime.min.time())
            current_day_end = current_day_start + timedelta(days=1)
            if start < current_day_end and end > current_day_start:
                seg_start = max(start, current_day_start)
                seg_end = min(end, current_day_end)
                ax.hlines(i,
                          (seg_start - current_day_start).total_seconds() / 3600,
                          (seg_end - current_day_start).total_seconds() / 3600,
                          colors='skyblue', linewidth=10, label='Сон' if i == 0 else "")

    # Смесь
    if not feed_df.empty:
        for _, row in feed_df[feed_df['date'] == day].iterrows():
            time_point = row['time']
            amount = row['amount']
            time_offset = (time_point - base_time).total_seconds() / 3600
            color = get_color_by_amount(amount)
            ax.plot(time_offset, i, 'o', color=color, markersize=8)

    # Ванна
    #if not bath_df.empty:
    #    for _, row in bath_df[bath_df['date'] == day].iterrows():
    #        time_offset = (row['time'] - base_time).total_seconds() / 3600
    #        ax.plot(time_offset, i, 'o', color='green', markersize=8)

    # Покакал
    #if not toilet_df.empty:
    #    for _, row in toilet_df[toilet_df['date'] == day].iterrows():
    #        time_offset = (row['time'] - base_time).total_seconds() / 3600
     #       ax.plot(time_offset, i, 'o', color='brown', markersize=2)

ax.set_yticks(range(len(all_dates)))
ax.set_yticklabels([day.strftime('%Y-%m-%d') for day in all_dates])
ax.set_xlabel('Время суток (часы)')
ax.set_xlim(0, 24)
ax.invert_yaxis()
ax.grid(True, axis='x', linestyle='--', alpha=0.5)
ax.set_xticks(range(0, 24))
ax.set_xticklabels([f'{h}' for h in range(0, 24)])
plt.title('Сон, питание и события')
plt.tight_layout()
fig1.savefig("events_timeline.png", dpi=150)
plt.close()

# === 7. Второй график (суммарные значения) ===
if not sleep_df.empty:
    sleep_df['duration_min'] = (sleep_df['end'] - sleep_df['start']).dt.total_seconds() / 60
else:
    sleep_df['duration_min'] = pd.Series(dtype=float)

daily_sleep = sleep_df.groupby('date')['duration_min'].sum().reset_index() if not sleep_df.empty else pd.DataFrame(columns=['date', 'duration_min'])
daily_feed = feed_df.groupby('date')['amount'].sum().reset_index() if not feed_df.empty else pd.DataFrame(columns=['date', 'amount'])
merged = pd.merge(daily_feed, daily_sleep, on='date', how='outer').sort_values('date')
merged['duration_min'] = merged['duration_min'].fillna(0)
merged['amount'] = merged['amount'].fillna(0)
merged['duration_hr'] = merged['duration_min'] / 60

fig2, ax1 = plt.subplots(figsize=(10, 5))
color1 = 'tab:orange'
ax1.set_xlabel('Дата')
ax1.set_ylabel('Смесь (мл)', color=color1)
ax1.bar(merged['date'], merged['amount'], color=color1, alpha=0.6)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_ylim(0, max(merged['amount'].fillna(0).max(), 120))

ax2 = ax1.twinx()
color2 = 'tab:blue'
ax2.set_ylabel('Сон (часы)', color=color2)
ax2.plot(merged['date'], merged['duration_hr'], color=color2, marker='o')
ax2.tick_params(axis='y', labelcolor=color2)
ax2.set_ylim(0, max(merged['duration_hr'].fillna(0).max(), 15))
ax1.grid(axis='y', linestyle='--', alpha=0.5)
fig2.autofmt_xdate()
plt.title('Суммарная смесь и сон по дням')
fig2.tight_layout()
fig2.savefig("daily_summary.png", dpi=150)
plt.close()


# === 8. Новая логика: классификация дневного/ночного сна и расчёт времени бодрствования ===
#
# Правила (реализованы согласно описанным тобой условиям; см. комментарии ниже):
# - Дневной период: 06:00 - 20:00 (добавочное правило: любой сон, начавшийся между 09:00 и 17:00 — всегда дневной)
# - Если сон начался в интервале 06:00-20:00, но соседний (предыдущий или следующий) сон находится на расстоянии <= 15 минут,
#   то такой отрезок считаем ночным (т.е. короткие паузы <=25 мин объединяем к ночи).
# - Всё остальное — ночной сон.
# - Для расчёта по дате мы считаем пересечение каждого отрезка сна с календарными сутками 00:00-24:00.
#
# Замечание: правила исходят из твоего описания; при обнаружении неоднозначностей (например, очень короткие пересечения через полночь)
#   мы используем указанную выше логику классификации по моменту начала сна и соседним отрезкам.

# Параметры
DAY_START = time(6, 0)
DAY_END = time(20, 0)
FORCE_DAY_START = time(9, 0)
FORCE_DAY_END = time(17, 0)
GAP_THRESHOLD = timedelta(minutes=25)

# Подготовка списка сегментов сна
sleep_segments = []
if not sleep_df.empty:
    # сформируем список сегментов (start, end) и отсортируем
    for _, r in sleep_df[['start', 'end']].iterrows():
        sleep_segments.append({'start': r['start'], 'end': r['end']})
    sleep_segments = sorted(sleep_segments, key=lambda s: s['start'])

# классификация каждого сегмента как дневного/ночного
for idx, seg in enumerate(sleep_segments):
    start = seg['start']
    end = seg['end']
    st_time = start.time()

    # проверяем соседей
    prev_seg = sleep_segments[idx-1] if idx > 0 else None
    next_seg = sleep_segments[idx+1] if idx < len(sleep_segments)-1 else None

    adjacent_short = False
    if prev_seg is not None:
        # разрыв между концом предыдущего и началом текущего
        gap_prev = start - prev_seg['end']
        if gap_prev <= GAP_THRESHOLD:
            adjacent_short = True
    if next_seg is not None:
        gap_next = next_seg['start'] - end
        if gap_next <= GAP_THRESHOLD:
            adjacent_short = True

    # логика классификации
    if FORCE_DAY_START <= st_time <= FORCE_DAY_END:
        seg['is_day'] = True
    elif DAY_START <= st_time < DAY_END:
        # если соседние разрывы маленькие — считаем ночным, иначе дневным
        seg['is_day'] = not adjacent_short
    else:
        seg['is_day'] = False

# построчный расчёт по дням: дневной, ночной сон и бодрствование
per_day_stats = []
if sleep_segments or (not feed_df.empty):
    # диапазон дат: по снам и по кормлениям (чтобы не терять дни без сна)
    date_candidates = []
    if sleep_segments:
        date_candidates.extend([s['start'].date() for s in sleep_segments])
        date_candidates.extend([s['end'].date() for s in sleep_segments])
    if not feed_df.empty:
        date_candidates.extend(list(feed_df['date'].unique()))
    min_date = min(date_candidates)
    max_date = max(date_candidates)
    date_range = pd.date_range(min_date, max_date).date
else:
    date_range = []

for day in date_range:
    day_start = datetime.combine(day, datetime.min.time())
    day_end = day_start + timedelta(days=1)

    day_sleep_min = 0.0
    night_sleep_min = 0.0

    # суммируем пересечения каждого сегмента со днём
    for seg in sleep_segments:
        seg_start = seg['start']
        seg_end = seg['end']
        overlap_start = max(seg_start, day_start)
        overlap_end = min(seg_end, day_end)
        if overlap_end > overlap_start:
            overlap_min = (overlap_end - overlap_start).total_seconds() / 60.0
            if seg.get('is_day', False):
                day_sleep_min += overlap_min
            else:
                night_sleep_min += overlap_min

    total_sleep_min = day_sleep_min + night_sleep_min
    awake_min = 24 * 60 - total_sleep_min
    # защита от отрицательных значений (на всякий случай)
    if awake_min < 0:
        awake_min = 0

    per_day_stats.append({
        'date': day,
        'day_sleep_min': round(day_sleep_min, 2),
        'night_sleep_min': round(night_sleep_min, 2),
        'awake_min': round(awake_min, 2)
    })

stats_df = pd.DataFrame(per_day_stats)
if not stats_df.empty:
    stats_df['day_sleep_hr'] = stats_df['day_sleep_min'] / 60.0
    stats_df['night_sleep_hr'] = stats_df['night_sleep_min'] / 60.0
    stats_df['awake_hr'] = stats_df['awake_min'] / 60.0

# сохраняем таблицу для отладки (опционально)
# stats_df.to_csv('sleep_day_night_awake.csv', index=False)

# === 9. Третий график: stacked horizontal bar chart (ночной, дневной, бодрствование) ===
if not stats_df.empty:
    fig3, ax3 = plt.subplots(figsize=(12, max(1, len(stats_df)) * 0.5))

    y_pos = np.arange(len(stats_df))
    bar_height = 0.6

    # Рисуем сегменты
    bars_night = ax3.barh(y_pos, stats_df['night_sleep_hr'], color='purple', height=bar_height, label='Ночной сон')
    bars_day = ax3.barh(y_pos, stats_df['day_sleep_hr'], left=stats_df['night_sleep_hr'],
                        color='skyblue', height=bar_height, label='Дневной сон')
    bars_awake = ax3.barh(y_pos, stats_df['awake_hr'],
                          left=stats_df['night_sleep_hr'] + stats_df['day_sleep_hr'],
                          color='orange', height=bar_height, label='Бодрствование')

    # Подписи внутри сегментов
    def add_labels(bars, values):
        for bar, val in zip(bars, values):
            if val > 0.2:  # не писать на слишком маленьких сегментах
                ax3.text(bar.get_x() + bar.get_width() / 2,
                         bar.get_y() + bar.get_height() / 2,
                         f"{val:.1f}h",
                         ha='center', va='center', color='white', fontsize=9, fontweight='bold')

    add_labels(bars_night, stats_df['night_sleep_hr'])
    add_labels(bars_day, stats_df['day_sleep_hr'])
    add_labels(bars_awake, stats_df['awake_hr'])

    # Настройки осей
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels([d.strftime('%Y-%m-%d') for d in stats_df['date']])
    ax3.invert_yaxis()  # старые даты сверху
    ax3.set_xlabel('Часы (суммарно 24)')
    ax3.set_xlim(0, 24)
    ax3.legend(loc='upper right')
    ax3.grid(axis='x', linestyle='--', alpha=0.5)

    plt.title('Дневной сон / Ночной сон / Бодрствование')
    plt.tight_layout()
    fig3.savefig("day_night_awake.png", dpi=150)
    plt.close()

else:
    print("Нет данных для построения третьего графика (stats_df пустой).")

# === Конец ===

# === БЛОК: отчёт за последние 3 дня (HTML) ===
from datetime import date as _date, datetime as _datetime, time as _time

def _format_min_to_hrmin_local(m):
    """Человеко-читаемый формат: '1 час 20 минут', '20 минут', '2 часа'."""
    if m is None:
        m = 0
    m = int(round(float(m)))
    if m < 0:
        m = 0
    h = m // 60
    mi = m % 60
    parts = []
    if h:
        # склонение для часов
        if h % 10 == 1 and h % 100 != 11:
            hw = "час"
        elif h % 10 in (2, 3, 4) and not (12 <= h % 100 <= 14):
            hw = "часа"
        else:
            hw = "часов"
        parts.append(f"{h} {hw}")
    if mi:
        # склонение для минут
        if mi % 10 == 1 and mi % 100 != 11:
            mw = "минута"
        elif mi % 10 in (2, 3, 4) and not (12 <= mi % 100 <= 14):
            mw = "минуты"
        else:
            mw = "минут"
        parts.append(f"{mi} {mw}")
    if not parts:
        return "0 минут"
    return " ".join(parts)

def _get_bedtime_and_wakeup_for_day(day, segments):
    """
    По логике пользователя:
    - wakeup = последний конец ночного сегмента, который заканчивается в 00:00-12:00 текущего дня.
    - bedtime = первый старт ночного сегмента, который начинается в 18:00-23:59 этого же дня.
    Возвращает (bedtime_dt_or_None, wakeup_dt_or_None).
    """
    # ожидаем segments — список словарей с ключами 'start','end','is_day'
    day_start = _datetime.combine(day, _time(0, 0))
    morning_end = day_start + timedelta(hours=12)
    evening_start = _datetime.combine(day, _time(18, 0))
    day_end = _datetime.combine(day, _time(23, 59, 59))

    night_segs = [s for s in segments if not s.get('is_day', False)]
    if not night_segs:
        return None, None

    # Найти wakeup: те сегменты, у которых end в [00:00, 12:00]
    wake_candidates = [s for s in night_segs if s['end'] >= day_start and s['end'] <= morning_end]
    wake_dt = None
    if wake_candidates:
        wake_dt = max(wake_candidates, key=lambda s: s['end'])['end']

    # Найти bedtime: те сегменты, у которых start в [18:00, 23:59]
    bed_candidates = [s for s in night_segs if s['start'] >= evening_start and s['start'] <= day_end]
    bed_dt = None
    if bed_candidates:
        bed_dt = min(bed_candidates, key=lambda s: s['start'])['start']

    return bed_dt, wake_dt

# Подготавливаем данные (используем существующие переменные из основного скрипта)
today = _datetime.now().date()
target_days = [today, today - timedelta(days=1), today - timedelta(days=2)]

# stats_map в основном скрипте уже строился как соответствие date -> row, используем его, если есть
stats_map_local = {}
if 'stats_map' in globals() and isinstance(stats_map, dict):
    stats_map_local = stats_map
# локальная сортировка сегментов сна
_local_sleep_segments = sorted(sleep_segments, key=lambda s: s['start']) if 'sleep_segments' in globals() else []

html_parts = []
html_parts.append("""<html>
<head>
    <meta charset='utf-8'>
    <title>Report: last 3 days</title>
    <style>
        body {font-family: Arial, Helvetica, sans-serif; padding: 20px; background-color: #f5f5f5;}
        h1 {text-align: center; color: #333;}
        .cards-container {display: flex; gap: 20px; margin: 20px 0; flex-wrap: wrap;}
        .card {flex: 1; min-width: 300px; background: white; border-radius: 8px; padding: 18px;
               box-shadow: 0 2px 8px rgba(0,0,0,0.08);}
        .card h2 {margin-top: 0; color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 8px;}
        .card ul {list-style: none; padding: 0; margin: 8px 0;}
        .card ul li {padding: 6px 0; border-bottom: 1px solid #f0f0f0;}
        .card ul li:last-child {border-bottom: none;}
        .card ol {padding-left: 20px; margin: 8px 0;}
        .small {font-size: 0.95em; color: #555;}
        .graphs {margin-top: 20px;}
        .graphs img {max-width: 60%; height: auto; border: 1px solid #ddd; padding: 4px; margin-bottom: 12px; display: block;}
    </style>
</head>
<body>""")

html_parts.append("<h1>Отчёт — последние 3 дня</h1>")
html_parts.append("<div class='cards-container'>")

months = {
    1: 'января', 2: 'февраля', 3: 'марта', 4: 'апреля',
    5: 'мая', 6: 'июня', 7: 'июля', 8: 'августа',
    9: 'сентября', 10: 'октября', 11: 'ноября', 12: 'декабря'
}

for idx, day in enumerate(target_days):
    label = "сегодня" if idx == 0 else ("вчера" if idx == 1 else "позавчера")
    html_parts.append("<div class='card'>")
    html_parts.append(f"<h2>{label} {day.day} {months[day.month]}</h2>")

    # Отход ко сну и подъём по твоей логике (в рамках одного дня)
    bedtime_dt, wakeup_dt = _get_bedtime_and_wakeup_for_day(day, _local_sleep_segments)
    bedtime_txt = bedtime_dt.strftime("%H:%M") if bedtime_dt is not None else "—"
    wakeup_txt = wakeup_dt.strftime("%H:%M") if wakeup_dt is not None else "—"

    html_parts.append("<ul>")
    html_parts.append(f"<li><strong>Подъём:</strong> {wakeup_txt}</li>")
    html_parts.append("</ul>")

    # Дневные сны этого дня
    ds_segs = [s for s in _local_sleep_segments if s.get('is_day', False) and s['start'].date() == day]
    ds_segs = sorted(ds_segs, key=lambda s: s['start'])
    # ВБ между подъёмом и первым дневным сном
    if wakeup_dt and ds_segs:
        wb1_min = int((ds_segs[0]['start'] - wakeup_dt).total_seconds() / 60)
        if wb1_min > 0:
            wb1_txt = _format_min_to_hrmin_local(wb1_min)
            html_parts.append(f"<p>ВБ {wb1_txt}</p>")

    if ds_segs:
        html_parts.append("<p><strong>Дневные сны:</strong></p><ol>")
        for i, seg in enumerate(ds_segs, 1):
            start_txt = seg['start'].strftime("%H:%M")
            end_txt = seg['end'].strftime("%H:%M")
            dur_min = int((seg['end'] - seg['start']).total_seconds() / 60)
            dur_txt = _format_min_to_hrmin_local(dur_min)
            html_parts.append(f"<li>{start_txt}–{end_txt} ({dur_txt})</li>")
            # ВБ до следующего дневного сна
            if i < len(ds_segs):
                wb_min = int((ds_segs[i]['start'] - seg['end']).total_seconds() / 60)
                if wb_min < 0:
                    wb_min = 0
                wb_txt = _format_min_to_hrmin_local(wb_min)
                html_parts.append(f"<p class='small'>ВБ {wb_txt}</p>")
        html_parts.append("</ol>")
    else:
        html_parts.append("<p><em>Дневных снов не найдено</em></p>")

    # === Расчёт текущего ВБ для "сегодня" ===
    if label == "сегодня":
        now = _datetime.now()
        past_sleeps = [s for s in _local_sleep_segments if s['end'] <= now]
        if past_sleeps:
            last_sleep = max(past_sleeps, key=lambda s: s['end'])
            current_awake_min = int((now - last_sleep['end']).total_seconds() / 60)
            current_awake_txt = _format_min_to_hrmin_local(max(0, current_awake_min))
            # элемент с меткой времени конца последнего сна
            html_parts.append(
                f"<p><strong>Текущее ВБ:</strong> "
                f"<span id='current-awake' data-start='{last_sleep['end'].isoformat()}'>{current_awake_txt}</span></p>"
            )
        else:
            html_parts.append("<p><strong>Текущее ВБ:</strong> —</p>")

    # ВБ между последним дневным сном и отходом ко сну
    if bedtime_dt and ds_segs:
        wb2_min = int((bedtime_dt - ds_segs[-1]['end']).total_seconds() / 60)
        if wb2_min > 0:
            wb2_txt = _format_min_to_hrmin_local(wb2_min)
            html_parts.append(f"<p class='small'>ВБ {wb2_txt}</p>")
    html_parts.append("<ul>")
    html_parts.append(f"<li><strong>Отход ко сну:</strong> {bedtime_txt}</li>")
    html_parts.append("</ul>")
    # Общие показатели: используем stats_map_local если есть, иначе считаем по сегментам
    row = stats_map_local.get(day, None)
    if row is not None:
        day_sleep_min = float(row.get('day_sleep_min', 0.0))
        night_sleep_min = float(row.get('night_sleep_min', 0.0))
        total_sleep_min = day_sleep_min + night_sleep_min
        awake_min = float(row.get('awake_min', max(0.0, 24*60 - total_sleep_min)))
    else:
        # запасной подсчёт пересечений с сутками
        day_start_dt = _datetime.combine(day, _time(0, 0))
        day_end_dt = day_start_dt + timedelta(days=1)
        total_sleep_min = 0.0
        day_sleep_min = 0.0
        for s in _local_sleep_segments:
            overlap_start = max(s['start'], day_start_dt)
            overlap_end = min(s['end'], day_end_dt)
            if overlap_end > overlap_start:
                overlap_min = (overlap_end - overlap_start).total_seconds() / 60.0
                total_sleep_min += overlap_min
                if s.get('is_day', False):
                    day_sleep_min += overlap_min
        night_sleep_min = max(0.0, total_sleep_min - day_sleep_min)
        awake_min = max(0.0, 24*60 - total_sleep_min)

    total_sleep_txt = _format_min_to_hrmin_local(total_sleep_min)
    day_sleep_txt = _format_min_to_hrmin_local(day_sleep_min)
    night_sleep_txt = _format_min_to_hrmin_local(night_sleep_min)
    awake_txt = _format_min_to_hrmin_local(awake_min)



    html_parts.append("<ul>")
    html_parts.append(f"<li><strong>Общее время сна за сутки:</strong> {total_sleep_txt}</li>")
    html_parts.append(f"<li><strong>Продолжительность ночного сна:</strong> {night_sleep_txt}</li>")
    html_parts.append(f"<li><strong>Продолжительность дневных снов:</strong> {day_sleep_txt}</li>")
    html_parts.append(f"<li><strong>Суммарное время бодрствования:</strong> {awake_txt}</li>")

    html_parts.append("</ul>")

    html_parts.append("</div>")  # .card

html_parts.append("</div>")  # .cards-container

html_parts.append("<div class='graphs'>")
for imgname in ("events_timeline.png", "daily_summary.png", "day_night_awake.png"):
    if os.path.exists(imgname):
        html_parts.append(f"<img src='{imgname}' alt='{imgname}'>")
    else:
        html_parts.append(f"<p>Файл {imgname} не найден.</p>")
html_parts.append("</div>")
html_parts.append("""
<script>
  const vb = document.getElementById('current-awake');
  if (vb && vb.dataset.start) {
    const start = new Date(vb.dataset.start);
    function updateAwake() {
      const diffMin = Math.floor((Date.now() - start.getTime()) / 60000);
      const h = Math.floor(diffMin / 60);
      const m = diffMin % 60;
      let text = '';
      if (h > 0) text += h + (h === 1 ? ' час ' : (h < 5 ? ' часа ' : ' часов '));
      text += m + (m % 10 === 1 && m % 100 !== 11 ? ' минута' :
                   (m % 10 >= 2 && m % 10 <= 4 && (m % 100 < 10 || m % 100 >= 20) ? ' минуты' : ' минут'));
      vb.textContent = text.trim();
    }
    updateAwake();
    setInterval(updateAwake, 60000);
  }
</script>
""")
html_parts.append("</body></html>")

report_fn = "last3days_report.html"
with open(report_fn, "w", encoding="utf-8") as fh:
    fh.write("\n".join(html_parts))

print(f"Отчет сохранён: {report_fn}")
# === КОНЕЦ блока ===



print("Готово: сохранены файлы events_timeline.png, daily_summary.png, day_night_awake.png (если были данные).")
