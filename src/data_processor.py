import os
import re
import numpy as np
import pandas as pd

# Monthly climate averages for San Antonio high low mean and rain
SAN_ANTONIO_CLIMATE = {
    1: (63, 41, 52.0, 1.96),
    2: (68, 45, 56.5, 1.74),
    3: (74, 52, 63.0, 2.31),
    4: (80, 58, 69.0, 2.42),
    5: (87, 66, 76.5, 4.40),
    6: (92, 73, 82.5, 3.28),
    7: (95, 75, 85.0, 2.41),
    8: (96, 75, 85.5, 2.15),
    9: (90, 70, 80.0, 3.88),
    10: (82, 60, 71.0, 3.75),
    11: (72, 50, 61.0, 2.08),
    12: (65, 43, 54.0, 1.83),
}

class DataProcessor:
    def __init__(self, file_path="data/acs_data.csv"):
        self.file_path = file_path
        self.raw_df = None
        self.clean_df = None
        
        # Reference values for normalization
        self.max_weekly_calls = 100.0
        self.max_weekly_strays = 30.0
        
        self.load_data()
        self.clean_data()
        
    def load_data(self):
        """Loads raw animal care services data from CSV"""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Dataset not found at: {self.file_path}")
        print(f"Reading dataset: {self.file_path}")
        self.raw_df = pd.read_csv(self.file_path)
        print(f"Loaded {len(self.raw_df)} raw records.")

    def clean_data(self):
        """Cleans columns isolates districts and standardizes dates"""
        df = self.raw_df.dropna(subset=['OPENEDDATETIME', 'COUNCIL_DIST']).copy()
        
        # Convert district to integers 1 to 10
        df['COUNCIL_DIST'] = df['COUNCIL_DIST'].astype(str).str.extract(r'(\d+)').astype(float)
        df = df[df['COUNCIL_DIST'].isin(range(1, 11))].copy()
        df['COUNCIL_DIST'] = df['COUNCIL_DIST'].astype(int)
        
        # Parse dates
        df['OPENEDDATETIME'] = pd.to_datetime(df['OPENEDDATETIME'], errors='coerce')
        df = df.dropna(subset=['OPENEDDATETIME']).copy()
        
        # Exclude anomalies in closed dates
        df['CLOSEDDATETIME'] = pd.to_datetime(df['CLOSEDDATETIME'], errors='coerce')
        
        # Create temporal groups
        df['Year'] = df['OPENEDDATETIME'].dt.year
        df['Week_of_Year'] = df['OPENEDDATETIME'].dt.isocalendar().week.astype(int)
        df['Month'] = df['OPENEDDATETIME'].dt.month.astype(int)
        
        self.clean_df = df
        print(f"Cleaned and validated {len(self.clean_df)} records.")

    def extract_street_name(self, address):
        """Extract clean street name from address"""
        if pd.isna(address):
            return "Unknown Street"
        # Get primary address segment
        parts = address.split(",")
        street_part = parts[0].strip()
        # Remove house numbers
        street_name = re.sub(r'^\d+\s+', '', street_part)
        # Remove units or apartments
        street_name = re.sub(r'\s+(APT|STE|UNIT|BLDG)\s+.*$', '', street_name, flags=re.IGNORECASE)
        return street_name.strip()

    def get_district_insights(self, district_id):
        """Get top historical calls and locations for district context"""
        dist_df = self.clean_df[self.clean_df['COUNCIL_DIST'] == district_id]
        if dist_df.empty:
            return {
                "top_call_types": [],
                "top_locations": [],
                "total_historical_calls": 0
            }
            
        # Top three call types
        top_types = dist_df['TYPENAME'].value_counts().head(3).index.tolist()
        
        # Top three street names
        dist_df_copy = dist_df.copy()
        dist_df_copy['Street'] = dist_df_copy['OBJECTDESC'].apply(self.extract_street_name)
        top_streets = dist_df_copy['Street'].value_counts().head(3).index.tolist()
        
        return {
            "top_call_types": top_types,
            "top_locations": top_streets,
            "total_historical_calls": len(dist_df)
        }

    def preprocess_and_aggregate(self, augment=True, augmentation_factor=5, random_seed=42):
        """Aggregate data and apply weather augmentation to train models"""
        np.random.seed(random_seed)
        
        print("Aggregating base historical counts...")
        # Perform base aggregations
        base_grouped = self.clean_df.groupby(['Year', 'Month', 'Week_of_Year', 'COUNCIL_DIST']).agg(
            Weekly_Call_Count=('CASEID', 'count'),
            Weekly_Stray_Count=('TYPENAME', lambda x: (x == 'Animals(Stray Animal)').sum()),
            Weekly_Aggressive_Count=('TYPENAME', lambda x: (x.str.contains('Aggressive|Bite', case=False, na=False)).sum()),
            Weekly_Late_Count=('Late (Yes/No)', lambda x: (x == 'YES').sum())
        ).reset_index()
        
        # Ensure every week and district combo is represented
        all_weeks = base_grouped[['Year', 'Month', 'Week_of_Year']].drop_duplicates()
        all_districts = pd.DataFrame({'COUNCIL_DIST': range(1, 11)})
        grid_df = all_weeks.merge(all_districts, how='cross')
        
        final_base = grid_df.merge(
            base_grouped, 
            on=['Year', 'Month', 'Week_of_Year', 'COUNCIL_DIST'], 
            how='left'
        ).fillna(0)
        
        # Cast to integer types
        final_base = final_base.astype({
            'Weekly_Call_Count': int,
            'Weekly_Stray_Count': int,
            'Weekly_Aggressive_Count': int,
            'Weekly_Late_Count': int
        })
        
        # Establish normalization scales
        self.max_weekly_calls = max(1.0, final_base['Weekly_Call_Count'].quantile(0.95))
        self.max_weekly_strays = max(1.0, final_base['Weekly_Stray_Count'].quantile(0.95))
        
        records = []
        
        print(f"Applying weather data augmentation (factor = {augmentation_factor if augment else 1})...")
        # Apply weather mapping and augmentation
        for _, row in final_base.iterrows():
            year = int(row['Year'])
            month = int(row['Month'])
            week = int(row['Week_of_Year'])
            dist = int(row['COUNCIL_DIST'])
            
            c_base = row['Weekly_Call_Count']
            s_base = row['Weekly_Stray_Count']
            a_base = row['Weekly_Aggressive_Count']
            l_base = row['Weekly_Late_Count']
            
            # Get historical climate baseline
            _, _, temp_mean, precip_mean = SAN_ANTONIO_CLIMATE[month]
            
            iterations = augmentation_factor if augment else 1
            for i in range(iterations):
                if augment and iterations > 1:
                    # Add random weather variations
                    d_temp = np.random.normal(0, 4.0)
                    d_precip = np.random.normal(0, 1.2)
                else:
                    d_temp = 0.0
                    d_precip = 0.0
                    
                temp_act = temp_mean + d_temp
                precip_act = max(0.0, precip_mean + d_precip)
                
                # Apply weather multipliers
                f_temp_general = np.clip(1.0 + 0.015 * d_temp, 0.7, 1.3)
                f_temp_stray = np.clip(1.0 + 0.02 * d_temp, 0.7, 1.3)
                f_temp_agg = np.clip(1.0 + 0.025 * d_temp, 0.7, 1.4)
                
                # Apply rain multipliers
                f_precip_general = np.clip(1.0 - 0.03 * d_precip, 0.6, 1.2)
                f_precip_stray = np.clip(1.0 - 0.06 * d_precip, 0.5, 1.1)
                f_precip_agg = np.clip(1.0 - 0.02 * d_precip, 0.7, 1.1)
                
                # Calculate new counts
                c_aug = max(0, int(round(c_base * f_temp_general * f_precip_general)))
                s_aug = max(0, int(round(s_base * f_temp_stray * f_precip_stray)))
                a_aug = max(0, int(round(a_base * f_temp_agg * f_precip_agg)))
                
                # Calculate late counts based on weather
                f_late_weather = np.clip(1.0 + 0.01 * d_temp + 0.03 * d_precip, 0.8, 1.4)
                l_aug = min(c_aug, max(0, int(round(l_base * f_late_weather * (c_aug / (c_base + 1.0))))))
                
                # Calculate strain score between 0 and 100
                v_score = min(100.0, (c_aug / self.max_weekly_calls) * 100.0)
                s_score = min(100.0, (s_aug / self.max_weekly_strays) * 100.0)
                l_ratio = (l_aug / c_aug) * 100.0 if c_aug > 0 else 0.0
                
                # Compute weighted strain score
                css = round(0.4 * v_score + 0.3 * s_score + 0.3 * l_ratio, 2)
                
                records.append({
                    'Year': year,
                    'Month': month,
                    'Week_of_Year': week,
                    'COUNCIL_DIST': dist,
                    'Temperature': round(temp_act, 1),
                    'Precipitation': round(precip_act, 2),
                    'Weekly_Call_Count': c_aug,
                    'Weekly_Stray_Count': s_aug,
                    'Weekly_Aggressive_Count': a_aug,
                    'Weekly_Late_Count': l_aug,
                    'Capacity_Strain_Score': css
                })
                
        augmented_df = pd.DataFrame(records)
        print(f"Aggregation and augmentation complete. Final training set has {len(augmented_df)} records.")
        return augmented_df

    def get_weather_baseline(self, month):
        """Get baseline weather values for a given month"""
        if month in SAN_ANTONIO_CLIMATE:
            _, _, temp_mean, precip_mean = SAN_ANTONIO_CLIMATE[month]
            return temp_mean, precip_mean
        return 70.0, 2.5
