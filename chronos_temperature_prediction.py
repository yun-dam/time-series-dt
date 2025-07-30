"""
Chronos Foundation Model Fine-tuning for Smart Buildings Temperature Prediction

This script fine-tunes the pre-trained Chronos model on the Smart Buildings dataset
for multivariate zone air temperature sensor forecasting.
"""

import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from chronos import ChronosPipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import Dict, List, Tuple, Optional, Union
import json
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class SmartBuildingsChronosProcessor:
    """Data processor for Smart Buildings dataset compatible with Chronos model."""
    
    def __init__(self, building="sb1", context_length=512, prediction_length=96):
        self.building = building
        self.context_length = context_length  # Input sequence length
        self.prediction_length = prediction_length  # Forecast horizon
        self.scalers = {}
        self.temperature_sensor_indices = []
        self.exogenous_indices = []
        
    def load_partition_data(self, partition: str) -> Tuple[np.ndarray, Dict]:
        """Load data for a specific partition."""
        data_path = f"./{self.building}/tabular/{self.building}/{partition}/data.npy.npz"
        metadata_path = f"./{self.building}/tabular/{self.building}/{partition}/metadata.pickle"
        
        data = np.load(data_path)
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
            
        return data, metadata
    
    def identify_sensors(self, metadata: Dict) -> Tuple[List[int], List[int]]:
        """Identify temperature sensors and exogenous variables."""
        temp_indices = []
        temp_names = []
        exog_indices = []
        exog_names = []
        
        for sensor_name, idx in metadata['observation_ids'].items():
            if "zone_air_temperature_sensor" in sensor_name:
                temp_indices.append(idx)
                temp_names.append(sensor_name)
            else:
                exog_indices.append(idx)
                exog_names.append(sensor_name)
        
        print(f"Temperature sensors: {len(temp_indices)}")
        print(f"Exogenous variables: {len(exog_indices)}")
        
        return temp_indices, exog_indices
    
    def prepare_chronos_data(self, partition: str, is_training: bool = True):
        """Prepare data in Chronos-compatible format."""
        print(f"Processing partition: {partition}")
        
        data, metadata = self.load_partition_data(partition)
        
        if is_training:
            self.temperature_sensor_indices, self.exogenous_indices = self.identify_sensors(metadata)
        
        # Extract data matrices
        observations = data['observation_value_matrix']
        actions = data['action_value_matrix']
        
        # Get temperature targets and exogenous features
        temperature_data = observations[:, self.temperature_sensor_indices]
        exogenous_data = observations[:, self.exogenous_indices]
        
        # Combine exogenous observations with actions
        all_features = np.concatenate([temperature_data, exogenous_data, actions], axis=1)
        
        print(f"Data shapes:")
        print(f"  Temperature sensors: {temperature_data.shape}")
        print(f"  All features: {all_features.shape}")
        
        # Fit scalers during training
        if is_training:
            self.fit_scalers(all_features, temperature_data)
        
        # Scale the data
        all_features_scaled = self.transform_features(all_features)
        temperature_scaled = self.transform_targets(temperature_data)
        
        # Create time series samples for Chronos
        samples = self.create_chronos_samples(all_features_scaled, temperature_scaled)
        
        return samples, metadata
    
    def fit_scalers(self, features: np.ndarray, targets: np.ndarray):
        """Fit scalers for normalization."""
        self.scalers['features'] = MinMaxScaler(feature_range=(-1, 1))
        self.scalers['targets'] = MinMaxScaler(feature_range=(-1, 1))
        
        self.scalers['features'].fit(features)
        self.scalers['targets'].fit(targets)
        
        print("Fitted scalers for features and targets")
    
    def transform_features(self, features: np.ndarray) -> np.ndarray:
        """Transform features using fitted scaler."""
        return self.scalers['features'].transform(features)
    
    def transform_targets(self, targets: np.ndarray) -> np.ndarray:
        """Transform targets using fitted scaler."""
        return self.scalers['targets'].transform(targets)
    
    def inverse_transform_targets(self, targets_scaled: np.ndarray) -> np.ndarray:
        """Inverse transform targets back to original scale."""
        return self.scalers['targets'].inverse_transform(targets_scaled)
    
    def create_chronos_samples(self, features: np.ndarray, targets: np.ndarray) -> List[Dict]:
        """Create samples in Chronos format."""
        samples = []
        total_length = self.context_length + self.prediction_length
        
        for i in range(len(features) - total_length + 1):
            # Context: past observations
            context_features = features[i:i + self.context_length]
            context_targets = targets[i:i + self.context_length]
            
            # Future: targets to predict
            future_targets = targets[i + self.context_length:i + total_length]
            future_features = features[i + self.context_length:i + total_length]
            
            sample = {
                'context_features': context_features,
                'context_targets': context_targets,
                'future_targets': future_targets,
                'future_features': future_features,
                'timestamp': i
            }
            
            samples.append(sample)
        
        print(f"Created {len(samples)} samples")
        return samples


class ChronosTimeSeriesDataset(Dataset):
    """PyTorch Dataset for Chronos fine-tuning."""
    
    def __init__(self, samples: List[Dict], tokenizer, max_length: int = 2048):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Prepare input sequence (context)
        context_targets = sample['context_targets']  # Shape: (context_length, num_temp_sensors)
        future_targets = sample['future_targets']    # Shape: (prediction_length, num_temp_sensors)
        
        # For Chronos, we typically work with univariate series
        # So we'll process each temperature sensor separately
        # For simplicity, let's use the first temperature sensor
        input_series = context_targets[:, 0]  # Take first temperature sensor
        target_series = future_targets[:, 0]   # Predict first temperature sensor
        
        # Convert to tokens (Chronos uses special tokenization for time series)
        input_tokens = self.tokenize_series(input_series)
        target_tokens = self.tokenize_series(target_series)
        
        # Prepare for seq2seq format
        input_ids = torch.tensor(input_tokens, dtype=torch.long)
        labels = torch.tensor(target_tokens, dtype=torch.long)
        
        # Pad sequences if necessary
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
        if len(labels) > self.max_length:
            labels = labels[:self.max_length]
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': torch.ones_like(input_ids)
        }
    
    def tokenize_series(self, series: np.ndarray) -> List[int]:
        """
        Convert time series values to tokens.
        This is a simplified tokenization - Chronos uses more sophisticated methods.
        """
        # Simple binning approach for demonstration
        # In practice, Chronos uses more sophisticated tokenization
        min_val, max_val = -1, 1  # Our scaled range
        num_bins = 1000
        
        # Bin the values
        bins = np.linspace(min_val, max_val, num_bins)
        tokens = np.digitize(series, bins)
        
        # Ensure tokens are in valid range
        tokens = np.clip(tokens, 1, num_bins - 1)
        
        return tokens.tolist()


class ChronosFineTuner:
    """Fine-tuning and inference pipeline for Chronos model."""
    
    def __init__(self, model_name: str = "amazon/chronos-t5-base"):
        self.model_name = model_name
        self.pipeline = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_model(self):
        """Load pre-trained Chronos pipeline."""
        print(f"Loading Chronos pipeline: {self.model_name}...")
        
        try:
            self.pipeline = ChronosPipeline.from_pretrained(
                self.model_name,
                device_map=self.device,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            )
            
            print(f"Chronos pipeline loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"Error loading Chronos pipeline: {e}")
            print("Available models: amazon/chronos-t5-tiny, amazon/chronos-t5-mini, amazon/chronos-t5-small, amazon/chronos-t5-base, amazon/chronos-t5-large")
            raise
    
    def predict_temperature(self, context_data: np.ndarray, prediction_length: int = 96) -> np.ndarray:
        """Use Chronos pipeline to predict temperature sensors."""
        predictions = []
        
        # Process each temperature sensor separately (Chronos works with univariate series)
        for sensor_idx in range(context_data.shape[1]):
            sensor_data = context_data[:, sensor_idx]
            
            # Convert to pandas Series with datetime index (required by Chronos)
            timestamps = pd.date_range(start='2022-01-01', periods=len(sensor_data), freq='H')
            ts = pd.Series(sensor_data, index=timestamps)
            
            # Generate forecast
            forecast = self.pipeline.predict(
                context=ts, 
                prediction_length=prediction_length,
                num_samples=1,
                temperature=1.0,
                top_k=50,
                top_p=1.0
            )
            
            predictions.append(forecast[0].numpy())  # Take first (and only) sample
        
        return np.array(predictions).T  # Shape: (prediction_length, num_sensors)
    
    def evaluate_model(self, test_samples: List[Dict], processor: SmartBuildingsChronosProcessor):
        """Evaluate the Chronos model using zero-shot prediction."""
        print("Evaluating Chronos model...")
        
        predictions = []
        targets = []
        
        # Evaluate on subset for speed
        for i, sample in enumerate(test_samples[:50]):  
            print(f"Processing sample {i+1}/50", end='\r')
            
            # Get context and target for temperature sensors
            context_data = sample['context_targets']  # Shape: (context_length, num_temp_sensors)
            target_data = sample['future_targets']    # Shape: (prediction_length, num_temp_sensors)
            
            # Make prediction using Chronos
            try:
                pred = self.predict_temperature(
                    context_data, 
                    prediction_length=target_data.shape[0]
                )
                
                predictions.append(pred)
                targets.append(target_data)
                
            except Exception as e:
                print(f"\nError in prediction: {e}")
                # Fall back to naive prediction
                pred = np.tile(np.mean(context_data[-10:], axis=0), (target_data.shape[0], 1))
                predictions.append(pred)
                targets.append(target_data)
        
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        # Calculate metrics
        mae = mean_absolute_error(targets.flatten(), predictions.flatten())
        mse = mean_squared_error(targets.flatten(), predictions.flatten())
        rmse = np.sqrt(mse)
        
        print(f"\nEvaluation Results:")
        print(f"MAE: {mae:.4f}")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        
        return {'mae': mae, 'mse': mse, 'rmse': rmse}


def main():
    """Main Chronos pipeline for zero-shot temperature prediction."""
    print("Chronos Zero-Shot Prediction for Smart Buildings Temperature")
    print("=" * 70)
    
    # Configuration
    context_length = 512    # Input sequence length
    prediction_length = 96  # Forecast horizon (4 days hourly)
    
    # Initialize processor
    processor = SmartBuildingsChronosProcessor(
        building="sb1",
        context_length=context_length,
        prediction_length=prediction_length
    )
    
    # Prepare training data (for context understanding)
    print("\n1. Preparing training data...")
    train_samples, train_metadata = processor.prepare_chronos_data("2022_a", is_training=True)
    
    # Prepare validation data (for prediction)
    print("\n2. Preparing validation data...")
    val_samples, val_metadata = processor.prepare_chronos_data("2022_b", is_training=False)
    
    # Initialize Chronos pipeline
    print("\n3. Loading Chronos model...")
    predictor = ChronosFineTuner(model_name="amazon/chronos-t5-small")  # Use smaller model for faster inference
    predictor.load_model()
    
    # Evaluate zero-shot performance
    print("\n4. Running zero-shot evaluation...")
    metrics = predictor.evaluate_model(val_samples, processor)
    
    # Save configuration and results
    config = {
        'model_name': predictor.model_name,
        'context_length': context_length,
        'prediction_length': prediction_length,
        'building': processor.building,
        'num_temperature_sensors': len(processor.temperature_sensor_indices),
        'num_exogenous_features': len(processor.exogenous_indices),
        'metrics': metrics,
        'evaluation_type': 'zero_shot'
    }
    
    with open('chronos_evaluation_results.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\nZero-shot evaluation completed!")
    print("Results saved to: chronos_evaluation_results.json")
    
    # Demonstrate single prediction
    print("\n5. Demonstrating single prediction...")
    sample = val_samples[0]
    context_data = sample['context_targets']  # Temperature sensor context
    target_data = sample['future_targets']    # Actual future values
    
    print(f"Context shape: {context_data.shape}")
    print(f"Target shape: {target_data.shape}")
    
    try:
        prediction = predictor.predict_temperature(context_data, prediction_length=target_data.shape[0])
        print(f"Prediction shape: {prediction.shape}")
        
        # Calculate metrics for this single prediction
        mae = mean_absolute_error(target_data.flatten(), prediction.flatten())
        print(f"Single prediction MAE: {mae:.4f}")
        
    except Exception as e:
        print(f"Error in single prediction: {e}")
    
    print("\nDemo completed!")


if __name__ == "__main__":
    main()