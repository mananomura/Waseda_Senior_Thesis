import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv(
    "/Local/NewProductSyntheticData.csv",
    sep=";",
    encoding="latin1"
)

# ---------------------------
# 1. Basic structure
# ---------------------------
print(df.shape)
print(df.dtypes)

# ---------------------------
# 2. Total demand distribution
# ---------------------------
plt.figure()
sns.histplot(df["TotalDemand"], bins=40, kde=True)
plt.title("Distribution of Total Demand")
plt.xlabel("Total Demand")
plt.ylabel("Frequency")
plt.show()

# ---------------------------
# 3. Price vs Total Demand
# ---------------------------
plt.figure()
sns.scatterplot(x=df["Price"], y=df["TotalDemand"], alpha=0.5)
plt.title("Price vs Total Demand")
plt.xlabel("Price")
plt.ylabel("Total Demand")
plt.show()

# ---------------------------
# 4. Average demand trajectory
# ---------------------------
demand_cols = [f"Demand{i:02d}" for i in range(1, 19)]
avg_demand = df[demand_cols].mean()

plt.figure()
plt.plot(range(1, 19), avg_demand)
plt.title("Average Weekly Demand Trajectory")
plt.xlabel("Week Since Launch")
plt.ylabel("Average Demand")
plt.show()
