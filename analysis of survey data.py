import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("simulated_ohc_data.csv")

# Convert Points ranges to numerical midpoints
points_mapping = {
    "0–100": 50,
    "101–300": 200.5,
    "301–500": 400.5
}
df["Points"] = df["Points"].map(points_mapping)

# Encode categorical variables
df["Feedback_Given"] = df["Feedback_Given"].map({"Yes": 1, "No": 0})
df["Health_Literacy"] = df["Health_Literacy"].map({"Low": 0, "Medium": 1, "High": 2})

# Drop UserID (not needed for analysis)
df = df.drop("UserID", axis=1)


print("Missing values:\n", df.isnull().sum())


# Plot distributions
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Badges Earned
sns.histplot(df["Badges_Earned"], bins=10, ax=axes[0, 0])
axes[0, 0].set_title("Distribution of Badges Earned")

# Posts per Week
sns.histplot(df["Posts_Week"], bins=15, ax=axes[0, 1])
axes[0, 1].set_title("Distribution of Weekly Posts")

# Health Literacy
sns.countplot(x=df["Health_Literacy"], ax=axes[1, 0])
axes[1, 0].set_title("Health Literacy Levels")

# Feedback Given
sns.countplot(x=df["Feedback_Given"], ax=axes[1, 1])
axes[1, 1].set_title("Feedback Given (Yes=1, No=0)")

plt.tight_layout()
plt.savefig("distributions.png")

# Correlation heatmap
plt.figure(figsize=(8, 6))
corr = df.corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.savefig("correlation_heatmap.png")



from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# Features and target
X = df[["Points", "Posts_Week", "Feedback_Given", "Health_Literacy"]]
y = df["Badges_Earned"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize and train model
model = XGBRegressor(objective="reg:squarederror", n_estimators=100)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Calculate RMSE
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"RMSE: {rmse:.2f} badges")

# Feature Importance
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)

print("\nFeature Importance:\n", importance)

# Create interaction term
df["Posts_x_Literacy"] = df["Posts_Week"] * df["Health_Literacy"]

# Re-run model with interaction
X = df[["Points", "Posts_Week", "Health_Literacy", "Posts_x_Literacy"]]
# ... repeat training/evaluation

# Relationship: Posts vs Badges by Literacy
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x="Posts_Week",
    y="Badges_Earned",
    hue="Health_Literacy",
    data=df,
    palette="viridis"
)
plt.title("Badges Earned vs Weekly Posts (Colored by Health Literacy)")
plt.savefig("posts_vs_badges.png")