import requests
import matplotlib.pyplot as plt
from datetime import datetime

# Step 1: Set up API details
API_KEY = 'YOUR_API_KEY_HERE'  # Replace with your actual API key
CITY = 'Delhi'
URL = f'https://api.openweathermap.org/data/2.5/forecast?q={CITY}&appid={API_KEY}&units=metric'

# Step 2: Fetch data from API
response = requests.get(URL)
data = response.json()

# Step 3: Extract time and temperature for next 24 hours (3-hour intervals)
times = []
temps = []

for item in data['list'][:8]:  # Get data for next 8 time slots (3h each = 24h)
    timestamp = datetime.fromtimestamp(item['dt']).strftime('%I:%M %p')
    temperature = item['main']['temp']
    times.append(timestamp)
    temps.append(temperature)

# Step 4: Visualize the data
plt.figure(figsize=(10, 5))
plt.plot(times, temps, color='skyblue', marker='o', linestyle='-', linewidth=2)
plt.title(f'Temperature Forecast for {CITY} (Next 24 Hours)', fontsize=14)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Temperature (°C)', fontsize=12)
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
