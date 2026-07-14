# San Antonio Animal Management: Backend Testing & UI Plan

This document outlines how to execute the backend AI pipeline and details the suggested layout structure for the final Streamlit UI dashboard.

---

## 🚀 Part 1: How to Run & Test the Backend

Ensure you are in the project root directory and your virtual environment is active:

```bash
# 1. Activate Virtual Environment
source venv/bin/activate

# 2. Run End-to-End Verification Pipeline
python app.py
```
*App.py executes the full ingestion, aggregation, model training, prediction, and Gemini LLM connection verification.*

### 🔮 Run Custom Scenario Testing (CLI tool)
The CLI tool `test_backend.py` allows you to customize the target scenario by providing any combination of inputs.

**Behavior & Dependencies:**
* **Independent Overrides**: You can specify **any single flag, any combination, or all flags** to run a prediction scenario.
* **Fallbacks**: Any flag you omit will fallback to its default baseline value. For example, if you provide `--temp 95.0` but omit `--precip`, the backend automatically resolves `--precip` using the historical baseline precipitation for the selected month.
* **No Mutual Exclusivity**: There are no dependencies forcing you to use all custom flags together.

#### Full CLI Command Flag Reference:
| Flag | Description | Valid Options / Range | Default (If Omitted) |
| :--- | :--- | :--- | :--- |
| `--train` | Force retraining of all 4 Random Forest models. | *(No value needed)* | Uses pre-trained pickled models from `/models` |
| `--district` | Target Council District for prediction. | `1` to `10` | `3` |
| `--month` | Target month of the year. | `1` to `12` | `7` (July) |
| `--week` | Target week of the year. | `1` to `53` | `28` |
| `--temp` | Customized temperature override in °F. | Float value (e.g. `95.0`) | Monthly climate average (e.g., `85.0°F` in July) |
| `--precip` | Customized precipitation override in inches. | Float value (e.g. `0.0`) | Monthly climate average (e.g., `2.41 in` in July) |

#### Usage Examples:
* **Test only a custom District** (uses baseline July weather defaults):
  ```bash
  python test_backend.py --district 2
  ```
* **Test only a custom Temperature** (uses District 3, July defaults for everything else):
  ```bash
  python test_backend.py --temp 102.0
  ```
* **Force Retraining + Run Custom Scenario**:
  ```bash
  python test_backend.py --train --district 5 --month 10 --week 42 --temp 95.0 --precip 0.0
  ```

---

## 🎨 Part 2: Suggested Streamlit UI Layout

To connect this backend to a user interface in the next phase, we suggest the following visual layout:

### 1. Sidebar: Scenario Controls
* **Target Location**: A dropdown select box for **City Council District (1-10)**.
* **Target Timeframe**: A slider for **Month of Year (1-12)**.
* **Weather Forecast Sliders**:
  * **Temperature Slider**: Slider from `40°F` to `110°F` (auto-filled with the seasonal baseline default when the Month is changed).
  * **Precipitation Slider**: Slider from `0.0 in` to `8.0 in` (auto-filled with seasonal baseline default).
* **"Generate Memo" Action Button**: Standard Streamlit submit button to trigger predictions and LLM calls.

### 2. Main Dashboard Panel
* **Top Row: Key Performance Indicators (KPIs)**:
  * Display three column cards using `st.metric()`:
    1. **Projected Total Cases**
    2. **Projected Stray Reports**
    3. **Projected Aggressive Incident Reports**
* **Second Row: Alert Banner (Capacity Warning)**:
  * A color-coded warning alert based on the predicted **Capacity Strain Score**:
    * `< 40`: Green indicator (Low Strain)
    * `40 - 75`: Orange indicator (Medium Strain)
    * `> 75`: Red indicator (High Strain - Action Required)
* **Third Row: Historical Context Comparison**:
  * Display a simple bar/line chart using `st.bar_chart` showing historical call volumes for the selected district vs. the current week's prediction.
  * Print historical hotspots (common streets) and top 3 call types.
* **Fourth Row: Copy-Pasteable Operational Memo**:
  * Render the generated operational memo inside a Markdown container (`st.markdown(results['operational_memo'])`).
  * Include a button to copy the text to the clipboard.
