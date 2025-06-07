#!/usr/bin/env python3
"""
End-to-End EDA & Feature Engineering Script for Acme Security Tickets
This script loads the CSV, performs exploratory data analysis (EDA),
generates visualizations, engineers features, computes correlations, and
prepares the dataset for modeling.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud

# ------------------------
# 1. CONFIGURATION
# ------------------------
CSV_PATH = r"data\acme_security_tickets.csv"  # adjust path as needed

# ------------------------
# 2. LOAD DATA
# ------------------------
df = pd.read_csv(CSV_PATH)

# ------------------------
# 3. INITIAL INSPECTION
# ------------------------
print("\n=== First 5 Rows of the DataFrame ===")
print(df.head())

print("\n=== DataFrame Info ===")
df.info()

print("\n=== Statistical Summary (Including Object Columns) ===")
print(df.describe(include="all"))

# Count missing values per column
missing_counts = df.isnull().sum().sort_values(ascending=False)
print("\n=== Missing Values per Column ===")
print(missing_counts)

# ------------------------
# 4. VALUE COUNTS FOR CATEGORICAL COLUMNS
# ------------------------
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

for col in cat_cols:
    counts = df[col].value_counts().to_frame(name="count")
    print(f"\n--- Value Counts for {col} ---")
    print(counts)

# ------------------------
# 5. DATA CLEANING & TYPE CONVERSIONS
# ------------------------
# Convert created_at to datetime (coerce errors to NaT)
df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")

# ------------------------
# 6. FEATURE ENGINEERING
# ------------------------
# 6.1 Time-based features
df["hour"] = df["created_at"].dt.hour
df["day_of_week"] = df["created_at"].dt.day_name()         # e.g., Monday, Tuesday
df["month"] = df["created_at"].dt.month                    # 1–12

# 6.2 Binary feature: were all mandatory fields provided?
def all_fields_check(row):
    mandatory = str(row["mandatory_fields"]).split("; ")
    provided = str(row["fields_provided"]).split("; ") if pd.notnull(row["fields_provided"]) else []
    return set(mandatory).issubset(set(provided))

df["all_fields_provided"] = df.apply(all_fields_check, axis=1)

# 6.3 Text length features
df["summary_length"] = df["request_summary"].fillna("").apply(len)
df["details_length"] = df["details"].fillna("").apply(len)

# 6.4 TF-IDF Feature Extraction and Analysis
# Extract TF-IDF features from request_summary
print("\n=== TF-IDF Feature Analysis ===")
tfidf = TfidfVectorizer(stop_words="english", max_features=100)
tfidf_matrix = tfidf.fit_transform(df["request_summary"].fillna(""))

# Get feature names and create a DataFrame of TF-IDF scores
feature_names = tfidf.get_feature_names_out()
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)

# Calculate mean TF-IDF score for each term
mean_tfidf = tfidf_df.mean().sort_values(ascending=False)

print("\nTop 10 Most Important Terms (by mean TF-IDF score):")
print(mean_tfidf.head(10))

# Visualize top terms
# Create a color map based on request types
request_types = df['request_type'].unique()
colors = plt.cm.viridis(np.linspace(0, 1, len(request_types)))
color_map = dict(zip(request_types, colors))

plt.figure(figsize=(12, 6))
bars = mean_tfidf.head(20).plot(kind='bar')

# Color each bar based on most common request type for that term
for idx, term in enumerate(mean_tfidf.head(20).index):
    # Get documents containing this term
    mask = tfidf_df[term] > 0
    # Find most common request type for documents with this term
    most_common_type = df.loc[mask, 'request_type'].mode()[0]
    bars.patches[idx].set_facecolor(color_map[most_common_type])

plt.title('Top 20 Terms by Mean TF-IDF Score (Colored by Most Common Request Type)')
plt.xlabel('Terms')
plt.ylabel('Mean TF-IDF Score')
plt.xticks(rotation=45, ha='right')

# Add legend
legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color) 
                  for color in color_map.values()]
plt.legend(legend_elements, color_map.keys(), 
          title='Request Types', bbox_to_anchor=(1.05, 1))

plt.tight_layout()
plt.show()
tfidf = TfidfVectorizer(stop_words="english", max_features=100)
tfidf_matrix = tfidf.fit_transform(df["request_summary"].fillna(""))

# ------------------------
# 7. CORRELATION ANALYSIS
# ------------------------
# 7.1 One-hot encode categorical columns (for correlation)
onehot_cols = ["requester_department", "requester_title", "request_type", "approver_role", "outcome"]
df_encoded = pd.get_dummies(df[onehot_cols], drop_first=False)

# 7.2 Add numeric columns for correlation
df_encoded["security_risk_score"] = df["security_risk_score"]
df_encoded["resolution_time_hours"] = df["resolution_time_hours"].fillna(0)  # fill NaN with 0 for correlation

# 7.3 Compute the full correlation matrix
corr_matrix = df_encoded.corr()

# 7.4 Plot global correlation heatmap
plt.figure(figsize=(18, 14))
sns.heatmap(corr_matrix, cmap="coolwarm", center=0, vmin=-1, vmax=1, 
            cbar_kws={"shrink": 0.5})
plt.title("Global Correlation Heatmap (One-Hot Encoded + Numeric)")
plt.tight_layout()
plt.show()

# ------------------------
# 8. ADDITIONAL VISUALIZATIONS & INSIGHTS
# ------------------------
# 8.1 Outcome distribution
plt.figure(figsize=(6, 4))
sns.countplot(x="outcome", data=df, palette="viridis")
plt.title("Outcome Distribution")
plt.xlabel("Outcome")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# 8.2 Security Risk Score by Outcome (boxplot)
plt.figure(figsize=(6, 4))
sns.boxplot(x="outcome", y="security_risk_score", data=df, palette="Set2")
plt.title("Security Risk Score by Outcome")
plt.xlabel("Outcome")
plt.ylabel("security_risk_score")
plt.tight_layout()
plt.show()

# 8.3 Summary Length by Outcome (boxplot)
plt.figure(figsize=(6, 4))
sns.boxplot(x="outcome", y="summary_length", data=df, palette="Set3")
plt.title("Request Summary Length by Outcome")
plt.xlabel("Outcome")
plt.ylabel("summary_length")
plt.tight_layout()
plt.show()

# 8.4 Word Cloud of Request Summaries
all_summaries = " ".join(df["request_summary"].fillna("").tolist())
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(all_summaries)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud of Request Summaries")
plt.tight_layout()
plt.show()

# 8.5 Request Type by Outcome (count plot)
plt.figure(figsize=(10, 5))
sns.countplot(data=df, x="request_type", hue="outcome", palette="bright")
plt.title("Request Type by Outcome")
plt.xlabel("request_type")
plt.ylabel("Count")
plt.xticks(rotation=45, ha="right")
plt.legend(title="Outcome")
plt.tight_layout()
plt.show()

# ------------------------
# 9. MISSINGNESS HEATMAP (if missingno is installed)
# ------------------------
try:
    import missingno as msno
    plt.figure(figsize=(8, 4))
    msno.matrix(df)
    plt.title("Missing Data Heatmap")
    plt.tight_layout()
    plt.show()
except ImportError:
    print("\n[Skipping missingno heatmap: 'missingno' library not installed]\n")

# ------------------------
# 10. SAVE PROCESSED DATA FOR MODELING
# ------------------------
# Select final feature set (example)
final_features = [
    "requester_department",
    "requester_title",
    "request_type",
    "approver_role",
    "security_risk_score",
    "resolution_time_hours",
    "hour",
    "day_of_week",
    "month",
    "all_fields_provided",
    "summary_length",
    "details_length",
    "outcome"
]

df_final = df[final_features].copy()

# Optionally, convert categorical features to numeric labels or keep as-is for one-hot later
# For example:
# from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()
# df_final["requester_department_enc"] = le.fit_transform(df_final["requester_department"])
# ...

# Save to a new CSV for modeling
OUTPUT_CSV = "acme_tickets_processed.csv"
df_final.to_csv(OUTPUT_CSV, index=False)
print(f"\nProcessed dataset saved to: {OUTPUT_CSV}")

# ------------------------
# 11. BASELINE MODEL (Optional)
#     You can uncomment and adjust the following section to fit a simple baseline.
# ------------------------
"""
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# 11.1 Prepare X and y
X = df_final.drop(columns=["outcome"])
y = df_final["outcome"]

# 11.2 One-hot encode categorical features
categorical_feats = ["requester_department", "requester_title", "request_type", 
                     "approver_role", "day_of_week"]
X_encoded = pd.get_dummies(X, columns=categorical_feats, drop_first=True)

# 11.3 Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

# 11.4 Fit a Logistic Regression Baseline
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# 11.5 Evaluate
y_pred = clf.predict(X_test)
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))
"""
# ------------------------
# END OF SCRIPT
# ------------------------
