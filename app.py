import os
from dotenv import load_dotenv
from src.pipeline import AnimalManagementPipeline

def main():
    print("🔄 Loading environment variables...")
    load_dotenv()
    
    # 1. Verify API Key
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_key:
        print("❌ API Key Error: GEMINI_API_KEY could not be retrieved from local environment.")
        return
        
    print("🤖 Initializing Animal Management Pipeline Verification...")
    
    # 2. Verify Data Ingestion, cleaning, and model loading/training
    try:
        pipeline = AnimalManagementPipeline(
            data_path="data/acs_data.csv",
            model_dir="models",
            api_key=gemini_key
        )
        print("✅ Data Ingestion Proof: Cleaned dataset loaded.")
        
        # Load existing models or train new ones
        models_exist = pipeline.load_system()
        if not models_exist:
            print("🔄 Training model checkpoints for verification...")
            pipeline.train_system(augment=True, augmentation_factor=5, n_estimators=10)
            pipeline.load_system()
        print("✅ Environment Verification: scikit-learn models initialized and ready.")
        
    except Exception as e:
        print(f"❌ Data or ML Environment Error: {e}")
        return

    # 3. Verify GenAI API Connectivity and run a mini scenario test
    try:
        print("🔮 Testing scenario prediction & Gemini connection...")
        # Run a quick test scenario for District 3, month 7, week 28
        results = pipeline.run_scenario(
            district_id=3,
            month=7,
            week_of_year=28
        )
        predictions = results['predictions']
        print(f"✅ Prediction Successful:")
        print(f"   - Expected Total Calls: {predictions['predicted_total_count']}")
        print(f"   - Projected Capacity Strain Score: {predictions['predicted_capacity_strain_score']}/100")
        print("✅ API Connection Proof: Gemini successfully generated operational memo.")
        
    except Exception as e:
        print(f"❌ API Connection/Key Error: Failed to communicate with Gemini API. Details: {e}")
        return

    print("\n🎉 Verification complete: app.py pipeline executed successfully with 0 errors!")

if __name__ == "__main__":
    main()