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
│   ├── acs_data.csv                  # Local historical 311 animal service calls dataset
│   └── Council_Districts.geojson     # San Antonio council district boundary data for map visualization
├── models/                           # Pre-trained Random Forest model files (.pkl)
├── src/                              # Source code modules
│   ├── data_processor.py             # Data processing and weather baseline logic
│   ├── model.py                      # Random Forest training and prediction wrapper
│   ├── llm_layer.py                  # Gemini LLM memo generation integration
│   └── pipeline.py                   # Master pipeline orchestration
├── dashboard.py                      # Interactive Streamlit dashboard application
├── app.py                            # Core pipeline and environment verification script
├── test_backend.py                   # CLI tool for custom scenario testing
├── requirements.txt                  # Required Python packages
├── .env                              # Local environment configurations (API keys)
├── .gitignore                        # Git ignore file
└── README.md                         # Project documentation
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
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Create a `.env` file in the root directory (if it doesn't already exist) and add your Gemini API Key:
```env
GEMINI_API_KEY=your_actual_api_key_here
```

### 4. Running the Streamlit Dashboard
Launch the web interface locally using Streamlit:
```bash
streamlit run dashboard.py
```
Once launched, the dashboard will open in your default browser (typically at `http://localhost:8501`).

### 🖥️ How to Use the Streamlit Interface

1. **Sidebar Scenario Controls**:
   * **Target City Council District**: Select a San Antonio Council District (Districts 1–10, labeled with geographic regions such as Downtown/Central, Eastside, Southside, etc.).
   * **Timeframe Sliders**: Choose the target **Month of Year** (1–12) and **Week of Year** (1–53).
   * **Weather Forecast Sliders**: Customize temperature (°F) and precipitation (inches) forecasts. These auto-populate with historical baseline weather averages whenever the target month is adjusted.
   * **Generate Memo Button**: Click to evaluate historical trends, run Random Forest predictions, and call Gemini 2.5 Flash to generate an operational action memo.
   * **Advanced Model Management (Expander)**: Retrain the backend Random Forest models on demand with custom augmentation factors and estimator tree counts.

2. **Dashboard Views & Outputs**:
   * **Projected Case Volumes (KPI Cards)**: Displays total predicted service call count, stray report count, and aggressive incident report count.
   * **Capacity Warning Level**: Color-coded alert banner reflecting the calculated **Capacity Strain Score** (0–100):
     * 🟢 **Low Strain (Score < 40)**: Normal operational thresholds.
     * 🟠 **Medium Strain (Score 40–75)**: Elevated volume; preemptive mobile clinic & foster coordination recommended.
     * 🔴 **High Strain (Score > 75)**: Critical risk of shelter capacity surge.
   * **Interactive District Map & Context**: Folium-powered interactive map rendering the selected district boundaries in high-contrast red alongside historical top call types and street hotspots.
   * **Operational Command & Early Warning Memo**: Structured administrative operational command plan generated by Gemini LLM, complete with an expandable raw Markdown block for easy copying.

### 5. Running the Backend Verification Script (Optional)
To verify the end-to-end data pipeline, model loading, and GenAI connectivity directly from the terminal:
```bash
python app.py
```

### 6. Custom CLI Scenario Testing (Optional)
You can also run custom scenario predictions directly from the command line using `test_backend.py`:
```bash
# Example: Run prediction for District 2 with customized high temperature
python test_backend.py --district 2 --temp 102.0
```

---

## 📊 Dataset Detail
The dataset is mapped to San Antonio Council Districts (1 through 10). The model aggregates 311 cases by:
* `Year`
* `Month`
* `Week_of_Year`
* `COUNCIL_DIST` (standardized to standard integers)

This structured matrix enables the Random Forest model to learn temporal and spatial patterns of animal-related service calls.
