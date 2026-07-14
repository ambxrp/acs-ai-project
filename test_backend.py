import os
import argparse
from dotenv import load_dotenv
from src.pipeline import AnimalManagementPipeline

def main():
    # Load environment variables
    load_dotenv()
    
    # Check for Gemini API key
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("⚠️ Warning: GEMINI_API_KEY is not set in your .env file or environment.")
        print("Generative memo creation will fail if a valid API key is not provided.")
        print("Please configure GEMINI_API_KEY in your .env file.")
        print("-" * 50)

    # Set up argument parsing
    parser = argparse.ArgumentParser(
        description="Test script for San Antonio Animal Management: Capacity & Stray Early Warning System Backend"
    )
    parser.add_argument("--train", action="store_true", help="Force retraining of the Random Forest models.")
    parser.add_argument("--district", type=int, default=3, choices=range(1, 11), help="Council District (1-10). Default is 3.")
    parser.add_argument("--month", type=int, default=7, choices=range(1, 12), help="Target month (1-12). Default is 7 (July).")
    parser.add_argument("--week", type=int, default=28, choices=range(1, 54), help="Target Week of Year (1-53). Default is 28.")
    parser.add_argument("--temp", type=float, help="Temperature override (°F). If omitted, uses historical baseline.")
    parser.add_argument("--precip", type=float, help="Precipitation override (inches). If omitted, uses historical baseline.")
    
    args = parser.parse_args()

    # Initialize Pipeline
    print("🤖 Initializing Animal Management Backend Pipeline...")
    pipeline = AnimalManagementPipeline(
        data_path="data/acs_data.csv",
        model_dir="models",
        api_key=api_key
    )

    # Train if forced or if model files don't exist
    models_exist = pipeline.load_system()
    if args.train or not models_exist:
        print("\n⚙️ Models need to be trained. Running training pipeline...")
        pipeline.train_system(
            augment=True,
            augmentation_factor=5,
            n_estimators=100
        )
        # Reload models
        pipeline.load_system()
    else:
        print("✅ Pre-trained models loaded successfully from disk.")

    # 1. Run Scenario
    print(f"\n🔮 Running predictive scenario for:")
    print(f"   - District: {args.district}")
    print(f"   - Month: {args.month} (Week {args.week})")
    
    # Print baseline weather for reference
    base_temp, base_precip = pipeline.data_processor.get_weather_baseline(args.month)
    print(f"   - Monthly Weather Baseline: Temp = {base_temp}°F, Precip = {base_precip} in")
    
    temp_in = args.temp if args.temp is not None else base_temp
    precip_in = args.precip if args.precip is not None else base_precip
    
    print(f"   - Simulated Scenario Weather: Temp = {temp_in}°F, Precip = {precip_in} in")
    print("-" * 50)

    # Check for API key before running scenario
    if not api_key:
        print("❌ Cannot run Gemini memo generation without an API key.")
        print("Please check your predictions below (computed via Random Forest Backend):")
        predictions = pipeline.model_manager.predict(
            district_id=args.district,
            month=args.month,
            week_of_year=args.week,
            temperature=temp_in,
            precipitation=precip_in
        )
        for key, val in predictions.items():
            print(f"   - {key}: {val}")
        return

    try:
        # Run entire backend pipeline
        results = pipeline.run_scenario(
            district_id=args.district,
            month=args.month,
            week_of_year=args.week,
            temp_override=args.temp,
            precip_override=args.precip
        )
        
        inputs = results['inputs']
        predictions = results['predictions']
        insights = results['insights']
        memo = results['operational_memo']
        
        print("\n🎯 Scenario Execution Output:")
        print("=" * 50)
        print("1. MODEL PREDICTIONS:")
        print(f"   - Expected Total Call Volume: {predictions['predicted_total_count']} calls")
        print(f"   - Predicted Stray/Roaming Volume: {predictions['predicted_stray_count']} calls")
        print(f"   - Predicted Aggressive/Bite Volume: {predictions['predicted_aggressive_count']} calls")
        print(f"   - Capacity Strain Score: {predictions['predicted_capacity_strain_score']}/100")
        print("-" * 50)
        print("2. HISTORICAL CONTEXT & INSIGHTS:")
        print(f"   - Total Historical Calls in District {args.district}: {insights['total_historical_calls']}")
        print(f"   - Top Call Types: {', '.join(insights['top_call_types'])}")
        print(f"   - Common Hotspot Streets: {', '.join(insights['top_locations'])}")
        print("=" * 50)
        print("\n3. GENERATED COMMAND MEMO:")
        print(memo)
        print("=" * 50)
        print("\n✅ Backend test executed successfully!")

    except Exception as e:
        print(f"❌ Error executing scenario: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
