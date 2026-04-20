from agent import FishingAgent
from visualization import plot_vessels

def main():
    agent = FishingAgent()

    gdf, explanation = agent.run("2025-01-01", "2025-01-15")
    m = plot_vessels(gdf)
    m.save("map.html")

    print(explanation)

if __name__ == "__main__":
    main()