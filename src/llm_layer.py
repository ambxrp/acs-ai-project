import os
from google import genai
from google.genai import types
import time
import dotenv

class LlmManager:
    def __init__(self, api_key=None, model_name=None):
        # Ensure environment variables from .env are loaded
        dotenv.load_dotenv()
        
        # Resolve API Key
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not set in the environment or passed arguments.")
            
        self.client = genai.Client(api_key=self.api_key)
        # Load model name from parameter or GEMINI_MODEL in environment (configured via .env file)
        # Defaults to "gemini-2.5-flash" if not specified or empty
        self.model_name = model_name or os.environ.get("GEMINI_MODEL") or "gemini-2.5-flash"

    def generate_operational_memo(self, inputs, predictions, insights):
        """
        Calls Gemini to generate a highly detailed operational memo based on 
        predictions and historical data context.
        """
        # Determine trends
        temp_val = inputs['temperature']
        precip_val = inputs['precipitation']
        
        # Simple descriptions of deviations
        temp_trend = "above historical baseline" if temp_val > inputs['baseline_temp'] + 2 else (
            "below historical baseline" if temp_val < inputs['baseline_temp'] - 2 else "normal seasonal range"
        )
        precip_trend = "wetter than usual" if precip_val > inputs['baseline_precip'] + 0.5 else (
            "drier than usual" if precip_val < inputs['baseline_precip'] - 0.5 else "normal seasonal precipitation"
        )
        
        months_map = {
            1: "January", 2: "February", 3: "March", 4: "April",
            5: "May", 6: "June", 7: "July", 8: "August",
            9: "September", 10: "October", 11: "November", 12: "December"
        }
        month_name = months_map.get(inputs['month'], f"Month {inputs['month']}")
        
        # Build prompt content
        prompt = f"""You are the San Antonio Animal Care Services (ACS) Predictive Command Assistant.
Your task is to generate a concise, high-impact, and action-oriented Operational Command & Early Warning Memo for municipal coordinators and field operations teams.
This memo must be BRIEF, highly direct, and free of unnecessary fluff. Write in an authoritative, administrative tone, keeping explanations to short paragraphs or a few bullet points.

### Administrative Details
- Target Council District: District {inputs['district_id']}
- Target Month: {month_name} (Targeting Week {inputs['week_of_year']})
- Forecasted Temperature: {temp_val}°F ({temp_trend}; baseline is {inputs['baseline_temp']}°F)
- Forecasted Precipitation: {precip_val} in ({precip_trend}; baseline is {inputs['baseline_precip']} in)

### ML Predictions for the Target Week
- Expected Total Call Volume: {predictions['predicted_total_count']} calls
- Expected Stray/Roaming Reports: {predictions['predicted_stray_count']} calls
- Expected Aggressive/Bite Reports: {predictions['predicted_aggressive_count']} calls
- Projected Capacity Strain Score: {predictions['predicted_capacity_strain_score']}/100

### Historical District Context
- Total historical cases in this district: {insights['total_historical_calls']}
- Top 3 historical call typenames: {", ".join(insights['top_call_types']) if insights['top_call_types'] else "N/A"}
- Top 3 historical incident streets (hotspots): {", ".join(insights['top_locations']) if insights['top_locations'] else "N/A"}

### Operational Parameters
- Standard Patrol Shift: 12-hour coverage, 10 active field officers
- Target Live-Release Rate: 90%
- Shelter Capacity Threshold: 85% occupancy (Strain Score > 75 indicates high risk of exceeding capacity)

Please generate the memo structured in Markdown with the following sections. Maintain maximum brevity (use short sentences, limit paragraphs to 2-3 sentences, or use 2-3 short bullet points per section):
1. **ADMINISTRATIVE ROUTING BLOCK**: Standard header (TO, FROM, DATE [July 14, 2026], SUBJECT: Operational Command & Early Warning Memo - District {inputs['district_id']}).
2. **EXECUTIVE RISK SUMMARY**: A clear assessment of the risk. State the overall threat level (LOW, MEDIUM, HIGH, or CRITICAL) based on the Capacity Strain Score ({predictions['predicted_capacity_strain_score']}) and explain why in 2 concise sentences.
3. **ML PREDICTION INTERPRETATION**: Briefly explain what the forecasted numbers indicate, and how the weather ({temp_val}°F, {precip_val} in rain) is expected to influence calls. (Limit to 1 short paragraph).
4. **TACTICAL FIELD OFFICER DEPLOYMENT**:
   - Provide 2-3 short, actionable bullet points for patrols.
   - Mandate prioritizing the hotspot streets: {", ".join(insights['top_locations']) if insights['top_locations'] else "N/A"}.
   - Keep weather safety warnings very brief.
5. **COMMUNITY MOBILIZATION & FOSTER OUTREACH**:
   - Provide 2 short bullet points for targeted marketing (e.g. heat/rain safety) and foster mobilization.
   - Mention the mobile clinic drive.
6. **RESOURCE & BACKLOG MITIGATION STEPS**:
   - Provide 2 short bullet points proposing operational adjustments based on the capacity strain score of {predictions['predicted_capacity_strain_score']}/100.
"""

        print(f"Calling Gemini API ({self.model_name}) to generate operational memo for District {inputs['district_id']}...")
        
        # Robust Retry Logic
        max_retries = 3
        retry_delay = 2 

        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.3
                    )
                )
                return response.text
                
            except Exception as e:
                error_msg = str(e)
                # If it's a 503, we wait and try again
                if "503" in error_msg and attempt < max_retries - 1:
                    print(f"⚠️ Gemini API busy (503). Retrying in {retry_delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                    retry_delay *= 2 
                else:
                    # For 401s, other errors, or if we run out of retries, return the graceful fallback
                    print(f"❌ LLM Generation Failed: {error_msg}")
                    return (
                        "### ⚠️ API UNAVAILABLE NOTICE\n\n"
                        "The Gemini AI Memo Generator is currently experiencing an issue and could not draft the memo "
                        "(this may be due to high server demand or an invalid API key).\n\n"
                        "**However, your local Machine Learning predictions executed successfully.** \n\n"
                        "Please review the projected weekly case volumes, historical data, and capacity strain warning "
                        "on the dashboard to coordinate your field deployments."
                    )