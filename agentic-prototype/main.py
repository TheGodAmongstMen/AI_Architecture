from agent import FishingAgent

agent = FishingAgent()

gdf, explanation = agent.run("2025-01-01", "2025-01-15")

print(explanation)

