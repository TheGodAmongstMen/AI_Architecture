from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
import numpy as np
import pandas as pd

class VesselAnalyzer:
    def detect_clusters(self, gdf):
        coords = np.array(list(zip(gdf.geometry.x, gdf.geometry.y)))

        clustering = DBSCAN(eps=0.1, min_samples=5).fit(coords)
        gdf["cluster"] = clustering.labels_

        return gdf

    def detect_anomalies(self, gdf):
        gdf["duration_hours"] = (
            pd.to_datetime(gdf["exit_timestamp"]) - 
            pd.to_datetime(gdf["entry_timestamp"])
        ).dt.total_seconds() / 3600

        features = gdf[["hours", "duration_hours"]].fillna(0)
        model = IsolationForest(contamination=0.05)
        gdf["anomaly"] = model.fit_predict(features)
        return gdf

    def loitering_score(self, gdf):
        return gdf.groupby("ssvid").size().sort_values(ascending=False)

        