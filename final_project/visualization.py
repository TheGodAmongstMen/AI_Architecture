import folium

def plot_vessels(gdf):
    m = folium.Map(
        location=[-10, -40],
        zoom_start=4,
        tiles="CartoDB positron"
    )

    for _, row in gdf.iterrows():
        # Default to "blue" if anomaly column is missing
        color = "red" if row.get("anomaly", 1) == -1 else "blue"

        # Ensure geometry exists
        if row.geometry is None:
            continue

        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=3,
            color=color,
            fill=True,
            fill_color=color
        ).add_to(m)

    return m