import os
from src.data_processor import DataProcessor
from src.model import ModelManager
from src.llm_layer import LlmManager

class AnimalManagementPipeline:
    def __init__(self, data_path="data/acs_data.csv", model_dir="models", api_key=None, model_name=None):
        self.data_path = data_path
        self.model_dir = model_dir
        self.api_key = api_key
        self.model_name = model_name
        
        # Initialize modules
        self.data_processor = DataProcessor(file_path=data_path)
        self.model_manager = ModelManager(model_dir=model_dir)
        self.llm_manager = None

    def train_system(self, augment=True, augmentation_factor=5, n_estimators=100, random_seed=42):
        """Run full data preparation and model training workflow"""
        print("Starting Pipeline Training")
        # Prepare training data
        training_df = self.data_processor.preprocess_and_aggregate(
            augment=augment, 
            augmentation_factor=augmentation_factor, 
            random_seed=random_seed
        )
        
        # Train model targets
        self.model_manager.train(training_df, n_estimators=n_estimators, random_seed=random_seed)
        
        # Save model state
        self.model_manager.save_models()
        print("Pipeline Training Complete")

    def load_system(self):
        """Load trained models from disk"""
        print("Loading model checkpoints from disk...")
        return self.model_manager.load_models()

    def run_scenario(self, district_id, month, week_of_year, temp_override=None, precip_override=None):
        """Run prediction scenario and generate memo"""
        # Ensure LLM Manager is initialized
        if self.llm_manager is None:
            self.llm_manager = LlmManager(api_key=self.api_key, model_name=self.model_name)
            
        # Resolve weather inputs
        base_temp, base_precip = self.data_processor.get_weather_baseline(month)
        
        temp_val = temp_override if temp_override is not None else base_temp
        precip_val = precip_override if precip_override is not None else base_precip
        
        # Run machine learning predictions
        predictions = self.model_manager.predict(
            district_id=district_id,
            month=month,
            week_of_year=week_of_year,
            temperature=temp_val,
            precipitation=precip_val
        )
        
        # Fetch historical insights
        insights = self.data_processor.get_district_insights(district_id)
        
        # Construct input details for LLM
        inputs = {
            'district_id': district_id,
            'month': month,
            'week_of_year': week_of_year,
            'temperature': temp_val,
            'precipitation': precip_val,
            'baseline_temp': base_temp,
            'baseline_precip': base_precip
        }
        
        # Generate command memo
        memo = self.llm_manager.generate_operational_memo(
            inputs=inputs,
            predictions=predictions,
            insights=insights
        )
        
        return {
            'inputs': inputs,
            'predictions': predictions,
            'insights': insights,
            'operational_memo': memo
        }
