# weather_dashboard.py

import requests
import matplotlib.pyplot as plt
from datetime import datetime

# --- User Configuration ---
API_KEY = 'YOUR_API_KEY_HERE'  # Replace with your OpenWeatherMap API Key
CITY = 'Delhi'
UNITS = 'metric'  # use 'imperial' for Fahrenheit

# --- Fetch weather forecast data ---
url = f"https://api.openweathermap.org/data/2.5/forecast?q={CITY}&appid={API_KEY}&units={UNITS}"
response = requests.get(url)
data = response.json()

# --- Parse required data (next 24 hours: 8 slots x 3 hours) ---
times = []
temps = []

for item in data['list'][:8]:
    timestamp = datetime.fromtimestamp(item['dt']).strftime('%a %I:%M %p')
    temperature = item['main']['temp']
    times.append(timestamp)
    temps.append(temperature)

# --- Visualization Dashboard using Matplotlib ---
plt.figure(figsize=(12, 6))
plt.plot(times, temps, marker='o', color='teal', linestyle='--', linewidth=2)

# --- Styling the Chart ---
plt.title(f"🌤️ Temperature Forecast for {CITY} (Next 24 Hours)", fontsize=16)
plt.xlabel("Time", fontsize=12)
plt.ylabel("Temperature (°C)", fontsize=12)
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

# --- Show the Dashboard ---
plt.show()
