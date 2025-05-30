# API-INTEGRATION-AND-DATA-VISUALIZATION

 COMPANY - CODETECH IT SOLUTION 
 NAME - MAYANK ARY
 INTERN ID - CT04DN1010
 DOMAIN - PYTHON
 DURATION - 4 WEEKS
 MENTOR - NEELA SANTOSH


Step 1: Understanding the Task
The task is to use Python to fetch live weather data from a public API and then visualize that data using a chart. In this project, we chose the OpenWeatherMap API as our data source. The idea is to get the forecasted temperature for a city over the next 24 hours and show that in a line graph using matplotlib.

Step 2: Getting the API Key
To use OpenWeatherMap, you need an API key. Here's how you get it:

Go to https://openweathermap.org/

Create a free account and log in

Navigate to “API Keys” in your profile

Copy your key — you’ll use it in your Python script

Step 3: Installing Required Python Libraries
You need to install the following libraries:

bash
Copy
Edit
pip install requests matplotlib
requests – used to connect to the API and get the data

matplotlib – used to create the line graph

datetime – to format the API’s time into readable form

Step 4: Connecting to the API
In your Python code:

You enter the city name and API key

A URL is formed to request weather data

The response comes in JSON format

You read that JSON and extract temperatures and times

Step 5: Extracting Relevant Data
OpenWeatherMap gives data for every 3 hours. That’s 8 data points for 24 hours.

We only extract:

Time of day for each forecast (like 3 PM, 6 PM…)

Temperature at that time

This gives us two lists: one for time, and one for temperature.

Step 6: Visualizing the Data
Using matplotlib, we:

Plot time on the x-axis

Plot temperature on the y-axis

Add markers (dots) and lines

Style the graph with titles, rotated labels, gridlines, and colors

Now we have a clean dashboard that shows temperature changes throughout the day.

Step 7: Interpreting the Output
The line graph shows:

Whether the temperature is going up or down

How quickly it’s changing

Helps users plan their day based on weather trends

For example, if the temperature is rising sharply, you know it’ll be hot later in the day.

Step 8: Why This Project is Useful
Shows how real-world APIs work

Helps practice JSON data handling

Teaches visualization techniques

Is practical — weather apps, smart home automation, or dashboards can use this logic

Scalable — can be expanded to include humidity, wind, rain, etc.

Step 9: Final Output
A working Python file that:

Automatically fetches fresh data

Generates a temperature vs. time graph

Runs in under 10 seconds and gives real insights

You can modify the script to use any city in the world — just change one variable.

Conclusion
This project is a simple yet powerful example of how Python, APIs, and data visualization come together. It introduces you to:

Working with web data

Parsing JSON

Using requests and matplotlib

Making data easy to understand for real-world users

By following each step, you’ve built your own weather dashboard — just like those in apps and websites, but coded by you!

