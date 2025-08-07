import pandas as pd
from faker import Faker
import random

fake = Faker()

# Generate 100 synthetic responses
data = []
for _ in range(100):
    user_id = fake.uuid4()[:8]  # Short UserID
    badges = random.randint(0, 10)  # 0–10 badges
    points = random.choice(["0–100", "101–300", "301–500"])
    posts_week = random.randint(0, 20)  # 0–20 posts/week
    feedback = random.choice(["Yes", "No"])
    health_literacy = random.choice(["Low", "Medium", "High"])

    data.append({
        "UserID": user_id,
        "Badges_Earned": badges,
        "Points": points,
        "Posts_Week": posts_week,
        "Feedback_Given": feedback,
        "Health_Literacy": health_literacy
    })

# Save to CSV
df = pd.DataFrame(data)
df.to_csv("simulated_ohc_data.csv", index=False)
print("Synthetic data generated!")