from gfw_client import GFWClient
from geo import GeoFilter
from analysis import VesselAnalyzer
from llm_agent import LLMAgent

class FishingAgent:
    def __init__(self):
        self.client = GFWClient()
        self.geo = GeoFilter()
        self.analyzer = VesselAnalyzer()
        self.llm = LLMAgent()
        self.history = []

    def run(self, start_date, end_date):
        raw = self.client.get_fishing_events(start_date, end_date)
        gdf = self.geo.to_gdf(raw["entries"])  

        gdf = self.analyzer.detect_clusters(gdf)
        gdf = self.analyzer.detect_anomalies(gdf)

        summary = gdf[["hours", "duration_hours"]].describe().to_dict()

        explanation = self.llm.explain_results({
            "anomalies": gdf[gdf["anomaly"] == -1].to_dict(orient="records"),
            "summary": summary
        })

        self.history.append({
            "time": (start_date, end_date),
            "summary": explanation
        })

        return gdf, explanation