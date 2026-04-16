import geopandas as gpd
import pandas as pd
import json

class GeoFilter:
    def __init__(self):
        pass

    def to_gdf(self, entries):
        rows = []
        for entry in entries:
            for dataset_key, records in entry.items():
                for r in records:
                    rows.append({
                        "lat": r.get("lat"),
                        "lon": r.get("lon"),
                        "hours": r.get("hours"),
                        "flag": r.get("flag"),
                        "geartype": r.get("geartype"),
                        "callsign": r.get("callsign"),
                        "imo": r.get("imo"),
                        "date": r.get("date"),
                        "entry_timestamp": r.get("entryTimestamp"),
                        "exit_timestamp": r.get("exitTimestamp"),
                    })

        df = pd.DataFrame(rows)

        if df.empty:
            return gpd.GeoDataFrame(columns=["lon", "lat", "geometry"], crs="EPSG:4326")

        return gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df.lon, df.lat),
            crs="EPSG:4326"
        )

    def filter_eez(self, gdf):
        if gdf.empty:
            return gdf

        # Ensure CRS matches
        gdf = gdf.to_crs("EPSG:4326")

        # Fast spatial filter using precomputed union
        return gdf[gdf.within(self.eez_union)]

    def get_api_geometry(self):
        """
        Returns simplified EEZ geometry for API queries
        """
        return self.eez_geojson