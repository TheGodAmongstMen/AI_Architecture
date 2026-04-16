# Overview
This system is designed to identify and explain potentially suspicious fishing activity using satellite-based data. It combines data collection, geospatial processing, machine learning, and basic language generation into one workflow that can be run automatically over a selected time period.

The process starts with the GFWClient, which pulls fishing activity data from the Global Fishing Watch API for a given date range and region. This data comes in a nested JSON format, so the GeoFilter class is used to clean and reorganize it into a GeoDataFrame. Each record is converted into a geographic point with associated information like time, vessel identifiers, and fishing effort.

Once the data is structured, the VesselAnalyzer handles the main analysis. It first applies DBSCAN clustering to group together nearby activity points, which can reveal areas where vessels are repeatedly operating. Then it calculates how long each fishing event lasts and uses an Isolation Forest model to detect anomalies. These anomalies represent vessels whose behavior differs from typical patterns, such as unusually long fishing durations or irregular activity levels.

After identifying these outliers, the system passes them to the LLMAgent, which generates short explanations describing why the vessels might be considered suspicious. Instead of just outputting raw data, this step helps make the results easier to understand by summarizing patterns like clustering, repeated presence, or unusual behavior.

The overall workflow is managed by the FishingAgent, which connects all the components and keeps a record of past runs. In the end, the system not only flags unusual fishing activity but also provides simple explanations, making it more useful for analysis or monitoring purposes.

# How to Use:
Add your Global Fishing Watch API token in `config_template.py`. Make sure you have all the necessary python packages (`pip install -r requirements.txt`) and run `python main.py`

