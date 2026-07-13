# San Antonio Animal Management: Capacity & Stray Early Warning System

> [NOTE]
> **Student Project:** This application was developed as a student project.

## 📌 Project Overview
This project addresses the civic optimization challenge of animal management in San Antonio. By combining traditional machine learning predictions with generative AI insights, it provides municipal coordinators with early warnings of shelter capacity spikes and stray animal hotspots. This helps proactively allocate resources, schedule mobile clinics, and coordinate foster networks, directly supporting the **Alamo City Resiliency** theme.

## ⚠️ Problem Domain & Justification
San Antonio faces a massive animal management challenge, characterized by:
* **Record Intake Rates:** Municipal shelters frequently experience capacity surges.
* **Persistent Strays:** High numbers of free-roaming and stray animal reports across multiple city council districts.
* **Operational Strain:** Capacity spikes create logistical friction, strain municipal funding, and threaten to drop the city's live-release metric below target thresholds.

By predicting capacity surges and incident spikes, the city can transform its response from **reactive** to **proactive**.

---

## 🗺️ The Blueprint
The system is built as a local **Streamlit dashboard** tailored for municipal coordinators.

### 📥 Inputs
Users configure scenarios via interactive Streamlit widgets:
* **Target Timeframe:** Month or upcoming season.
* **Weather Trends:** Forecasted temperature and precipitation levels.
* **Target Locations:** Specific San Antonio City Council Districts (Districts 1-10) or Zip Codes.

### ⚙️ Processing Pipeline
1. **Machine Learning Backend:** A Random Forest Regressor trained on historical 311 animal service calls estimates incoming stray-incident volume and capacity strain scores.
2. **Generative Layer:** The predictions and environmental context are passed to a Gemini LLM via the Google GenAI SDK to generate a structured operational command memo.

### 📤 Outputs
* **Interactive Data Visualizations:** Charts comparing historical trends against predicted risk scores.
* **Operational Action Plan:** An actionable, copy-pasteable administrative memo directing patrol officer focus and marketing campaigns for foster coordination.

---

## 🛠️ Technical Approach & Stack
This project follows a **non-agentic AI approach**, blending classic data science with modern GenAI:

* **Data Science Foundation:** 
  * `pandas` & `numpy` for data cleaning, transformation, and matrix building.
  * `scikit-learn` for training a `RandomForestRegressor`.
  * **Dataset:** Public [Open Data SA: 311 Service Calls Related to Animals](https://data.sanantonio.gov/) dataset.
* **Generative Layer:** 
  * Google GenAI SDK (`google-genai`) utilizing `gemini-2.5-flash` to craft command memos.
* **User Interface:** 
  * `streamlit` for the frontend.
* **Environment Management:**
  * `python-dotenv` for managing credentials safely.

---

## 📂 Project Structure
```
acs-project/
├── data/
│   └── acs_data.csv          # Local historical 311 animal service calls dataset
├── app.py                    # Main pipeline and environment validation script
├── .env                      # Local environment configurations (API keys)
├── .gitignore                # Git ignore file
└── README.md                 # Project documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.10+ installed.

### 2. Set Up Virtual Environment & Dependencies
Clone the repository and initialize a virtual environment:
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # On macOS/Linux
# venv\Scripts\activate   # On Windows

# Install required dependencies
pip install pandas numpy scikit-learn google-genai python-dotenv streamlit
```

### 3. Configure Environment Variables
Create a `.env` file in the root directory (if it doesn't already exist) and add your Gemini API Key:
```env
GEMINI_API_KEY=your_actual_api_key_here
```

### 4. Running the Verification Pipeline
The main `app.py` script acts as a test verification script for the data pipeline and GenAI connectivity:
```bash
python app.py
```
This script will:
* Load the local dataset from `data/acs_data.csv`.
* Clean and aggregate the service calls by year, week, and Council District.
* Verify your `scikit-learn` model initialization.
* Test communications with the Gemini API.

---

## 📊 Dataset Detail
The dataset is mapped to San Antonio Council Districts (1 through 10). The model aggregates 311 cases by:
* `Year`
* `Month`
* `Week_of_Year`
* `COUNCIL_DIST` (standardized to standard integers)

This structured matrix enables the Random Forest model to learn temporal and spatial patterns of animal-related service calls.
