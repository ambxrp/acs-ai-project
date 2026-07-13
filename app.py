import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from google import genai
from sklearn.ensemble import RandomForestRegressor

# Load environment variables from .env
load_dotenv()

# Instantiate Gemini Client
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# 1. Path to the local dataset in this repository
LOCAL_FILE_PATH = "data/acs_data.csv"

print(f"🔄 Reading local dataset from: {LOCAL_FILE_PATH}...")
try:
    raw_df = pd.read_csv(LOCAL_FILE_PATH)
    print(f"✅ Ingested {len(raw_df)} records from local storage.")
except FileNotFoundError:
    print(f"❌ Error: Could not find the file at {LOCAL_FILE_PATH}. Please verify it is in the data folder.")
    raise

# 2. Drop rows missing critical features (Dates or Districts)
# Note: The official Open Data SA schema uses 'OPENEDDATETIME' and 'COUNCIL_DIST'
clean_df = raw_df.dropna(subset=['OPENEDDATETIME', 'COUNCIL_DIST']).copy()

# 3. Clean and isolate the Council District numbers
# San Antonio uses Districts 1 through 10. We parse them cleanly as standard integers.
clean_df['COUNCIL_DIST'] = clean_df['COUNCIL_DIST'].astype(str).str.extract(r'(\d+)').astype(float)
clean_df = clean_df[clean_df['COUNCIL_DIST'].isin(range(1, 11))].astype({'COUNCIL_DIST': 'int'})

# 4. Standardize temporal features
clean_df['OPENEDDATETIME'] = pd.to_datetime(clean_df['OPENEDDATETIME'], errors='coerce')
clean_df = clean_df.dropna(subset=['OPENEDDATETIME'])

# Extract time groupings for scikit-learn training
clean_df['Year'] = clean_df['OPENEDDATETIME'].dt.year
clean_df['Week_of_Year'] = clean_df['OPENEDDATETIME'].dt.isocalendar().week
clean_df['Month'] = clean_df['OPENEDDATETIME'].dt.month

# 5. Aggregate: Group by Year, Month, Week_of_Year, and District to count call volumes
print("📊 Aggregating case volumes per district per week...")
features_df = clean_df.groupby(['Year', 'Month', 'Week_of_Year', 'COUNCIL_DIST']).size().reset_index(name='Weekly_Call_Count')

# 6. Ensure all districts are represented for every single week (filling gaps with 0 calls)
all_weeks = features_df[['Year', 'Month', 'Week_of_Year']].drop_duplicates()
all_districts = pd.DataFrame({'COUNCIL_DIST': range(1, 11)})

# Cross join to create a perfect grid of all possibilities
grid_df = all_weeks.merge(all_districts, how='cross')
final_training_df = grid_df.merge(features_df, on=['Year', 'Month', 'Week_of_Year', 'COUNCIL_DIST'], how='left').fillna(0)

# Sort for structure
final_training_df = final_training_df.sort_values(by=['Year', 'Week_of_Year', 'COUNCIL_DIST']).reset_index(drop=True)

print("\n✅ Matrix building complete! Ready for model training.")
print(final_training_df.head(10))


def test_pipeline():
    print("🚀 Starting local skeleton script verification...")

    # 1. Verify Data Ingestion Path
    local_path = "data/acs_data.csv"
    if os.path.exists(local_path):
        df = pd.read_csv(local_path, nrows=5)
        print(f"✅ Data Ingestion Proof: Successfully read local file. Columns found: {list(df.columns)}")
    else:
        print("❌ Data Ingestion Error: Local CSV file not found at path.")
        return

    # 2. Verify scikit-learn Environment
    try:
        model = RandomForestRegressor(n_estimators=10)
        print("✅ Environment Verification: scikit-learn modules initialized successfully.")
    except Exception as e:
        print(f"❌ ML Environment Error: {e}")
        return

    # 3. Retrieve API Key from Environment
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_key:
        print("❌ API Key Error: GEMINI_API_KEY could not be retrieved from local environment.")
        return

    # 4. Verify GenAI API Connectivity
    try:
        client = genai.Client(api_key=gemini_key)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents="Connection test. Respond with 'API Connected'."
        )
        print(f"✅ API Connection Proof: Gemini responded: '{response.text.strip()}'")
    except Exception as e:
        print(f"❌ API Key Error: Failed to communicate with Gemini API. Details: {e}")
        return

    print("\n🎉 Verification complete: Boilerplate script executed with 0 syntax errors!")


if __name__ == "__main__":
    test_pipeline()