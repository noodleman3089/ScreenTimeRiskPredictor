import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score
import joblib
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import logging
from dataclasses import dataclass


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration class for model parameters"""
    csv_file: str = 'Indian_Kids_Screen_Time.csv'
    test_size: float = 0.2
    random_state: int = 42
    max_iter: int = 1000
    model_dir: str = 'models'
    
    feature_columns = [
        'Avg_Daily_Screen_Time_hr',
        'Exceeded_Recommended_Limit',
        'Educational_to_Recreational_Ratio',
        'Age'
    ]
    
    health_impacts = {
        'poor_sleep': 'Poor Sleep',
        'eye_strain': 'Eye Strain', 
        'obesity_risk': 'Obesity Risk'
    }


class DataProcessor:
    """Handles data loading and preprocessing"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
    
    def load_data(self) -> pd.DataFrame:
        """Load data from CSV file"""
        try:
            df = pd.read_csv(self.config.csv_file)
            logger.info(f"Loaded data with shape: {df.shape}")
            return df
        except FileNotFoundError:
            logger.error(f"CSV file {self.config.csv_file} not found")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract and clean feature columns"""
        features = df[self.config.feature_columns].copy()
        
        # Clean numeric features
        numeric_columns = [
            'Avg_Daily_Screen_Time_hr',
            'Educational_to_Recreational_Ratio'
        ]
        
        for col in numeric_columns:
            features[col] = pd.to_numeric(features[col], errors='coerce').fillna(0.0)
        
        # Clean Age column
        features['Age'] = pd.to_numeric(features['Age'], errors='coerce').fillna(0).astype(int)
        
        # Clean boolean feature
        features['Exceeded_Recommended_Limit'] = self._normalize_boolean_column(
            features['Exceeded_Recommended_Limit']
        )
        
        logger.info("Features extracted and cleaned successfully")
        return features
    
    def extract_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract and process health impact labels"""
        # Process Health_Impacts column into binary labels
        impacts_lists = df['Health_Impacts'].apply(self._parse_health_impacts)
        
        labels = pd.DataFrame()
        for internal_name, display_name in self.config.health_impacts.items():
            labels[internal_name] = impacts_lists.apply(
                lambda lst: 1 if display_name in lst else 0
            )
        
        logger.info(f"Labels extracted: {list(labels.columns)}")
        return labels
    
    def _normalize_boolean_column(self, series: pd.Series) -> pd.Series:
        """Normalize boolean column to 0/1 integers"""
        mapping = {True: 1, False: 0, 'True': 1, 'False': 0}
        return series.map(mapping).fillna(0).astype(int)
    
    def _parse_health_impacts(self, impacts) -> List[str]:
        """Parse health impacts string into list"""
        if pd.isna(impacts):
            return []
        
        impacts_str = str(impacts).strip()
        if impacts_str == '' or impacts_str.lower() == 'none':
            return []
        
        return [impact.strip() for impact in impacts_str.split(',') if impact.strip()]


class HealthImpactModel:
    """Main model class for health impact prediction"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.data_processor = DataProcessor(config)
        self.scaler: Optional[StandardScaler] = None
        self.classifier: Optional[OneVsRestClassifier] = None
        self.model_dir = Path(config.model_dir)
        self.model_dir.mkdir(exist_ok=True)
    
    def prepare_data(self, features: pd.DataFrame, labels: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split and scale the data"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, 
            test_size=self.config.test_size, 
            random_state=self.config.random_state
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        logger.info(f"Data prepared - Train: {X_train_scaled.shape}, Test: {X_test_scaled.shape}")
        return X_train_scaled, X_test_scaled, y_train.values, y_test.values
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train the multi-label classification model"""
        base_classifier = LogisticRegression(
            max_iter=self.config.max_iter, 
            class_weight='balanced',
            random_state=self.config.random_state
        )
        
        self.classifier = OneVsRestClassifier(base_classifier)
        self.classifier.fit(X_train, y_train)
        logger.info("Model training completed")
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance"""
        if not self.classifier:
            raise ValueError("Model not trained yet")
        
        y_pred = self.classifier.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_micro = f1_score(y_test, y_pred, average='micro')
        
        # Print detailed classification report
        target_names = list(self.config.health_impacts.keys())
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=target_names))
        
        metrics = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_micro': f1_micro
        }
        
        logger.info(f"Model evaluation completed: {metrics}")
        return metrics
    
    def save_model(self) -> None:
        """Save trained model and scaler"""
        if not self.classifier or not self.scaler:
            raise ValueError("Model and scaler must be trained before saving")
        
        joblib.dump(self.classifier, self.model_dir / 'model.pkl')
        joblib.dump(self.scaler, self.model_dir / 'scaler.pkl')
        
        # Save config for reference
        config_dict = {
            'health_impacts': self.config.health_impacts,
            'feature_columns': self.config.feature_columns
        }
        joblib.dump(config_dict, self.model_dir / 'config.pkl')
        
        logger.info(f"Model saved to {self.model_dir}")
    
    def load_model(self) -> None:
        """Load trained model and scaler"""
        try:
            self.classifier = joblib.load(self.model_dir / 'model.pkl')
            self.scaler = joblib.load(self.model_dir / 'scaler.pkl')
            logger.info("Model loaded successfully")
        except FileNotFoundError:
            logger.error("Model files not found. Train the model first.")
            raise
    
    def train_full_pipeline(self) -> Dict[str, float]:
        """Complete training pipeline"""
        # Load and preprocess data
        df = self.data_processor.load_data()
        features = self.data_processor.extract_features(df)
        labels = self.data_processor.extract_labels(df)
        
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(features, labels)
        
        # Train model
        self.train(X_train, y_train)
        
        # Evaluate model
        metrics = self.evaluate(X_test, y_test)
        
        # Save model
        self.save_model()
        
        return metrics


class AdviceGenerator:
    """Generates personalized health advice based on predictions"""
    
    @staticmethod
    def generate_advice(predicted_labels: List[str]) -> List[str]:
        """Generate personalized advice based on predicted health risks"""
        advice_map = {
            'Poor Sleep': [
                "Establish a consistent bedtime routine for better sleep quality",
                "Avoid screens at least 1 hour before bedtime",
                "Use blue light filters on devices in the evening",
                "Create a dark, quiet sleep environment"
            ],
            'Eye Strain': [
                "Follow the 20-20-20 rule: Every 20 minutes, look at something 20 feet away for 20 seconds",
                "Adjust screen brightness to match your surroundings",
                "Take regular breaks from screen time every hour",
                "Ensure proper lighting to reduce glare"
            ],
            'Obesity Risk': [
                "Incorporate physical activity breaks during screen time",
                "Stand and move around every 30-60 minutes",
                "Balance screen time with outdoor activities",
                "Set daily screen time limits and stick to them"
            ]
        }
        
        advice = []
        for label in predicted_labels:
            if label in advice_map:
                advice.extend(advice_map[label])
        
        if not advice:
            advice.append("Great! No significant health risks detected. Keep maintaining healthy screen time habits!")
        
        return advice


class HealthPredictor:
    """High-level interface for making predictions"""
    
    def __init__(self, model_dir: str = 'models'):
        self.config = ModelConfig(model_dir=model_dir)
        self.model = HealthImpactModel(self.config)
        self.advice_generator = AdviceGenerator()
        self.model.load_model()
    
    def predict(self, features: Dict[str, float]) -> Tuple[List[str], List[str]]:
        """Make predictions and generate advice for new data"""
        # Prepare feature array
        feature_array = self._prepare_feature_array(features)
        
        # Scale features
        scaled_features = self.model.scaler.transform(feature_array)
        
        # Make prediction
        prediction = self.model.classifier.predict(scaled_features)
        predicted_indices = np.where(prediction[0] == 1)[0]
        
        # Convert to display labels
        internal_labels = list(self.config.health_impacts.keys())
        display_labels = [
            self.config.health_impacts[internal_labels[i]] 
            for i in predicted_indices
        ]
        
        # Generate advice
        advice = self.advice_generator.generate_advice(display_labels)
        
        return display_labels, advice
    
    def _prepare_feature_array(self, features: Dict[str, float]) -> np.ndarray:
        """Convert feature dictionary to numpy array in correct order"""
        # Support both snake_case and original column names
        key_mappings = {
            'avg_daily_screen_time_hr': 'Avg_Daily_Screen_Time_hr',
            'exceeded_recommended_limit': 'Exceeded_Recommended_Limit', 
            'educational_to_recreational_ratio': 'Educational_to_Recreational_Ratio',
            'age': 'Age'
        }
        
        def get_feature_value(possible_keys: List[str]) -> float:
            for key in possible_keys:
                if key in features:
                    return float(features[key])
            raise KeyError(f"None of keys {possible_keys} found in features")
        
        feature_values = []
        for col in self.config.feature_columns:
            snake_case_key = col.lower().replace(' ', '_')
            possible_keys = [snake_case_key, col]
            feature_values.append(get_feature_value(possible_keys))
        
        return np.array([feature_values])


def main():
    """Main function to train the model"""
    config = ModelConfig()
    model = HealthImpactModel(config)
    
    try:
        metrics = model.train_full_pipeline()
        print(f"\nModel training completed successfully!")
        print(f"Final metrics: {metrics}")
        
        # Example prediction
        print("\n" + "="*50)
        print("Testing prediction functionality...")
        
        predictor = HealthPredictor()
        test_features = {
            'avg_daily_screen_time_hr': 8.0,
            'exceeded_recommended_limit': 1,
            'educational_to_recreational_ratio': 0.3,
            'age': 12
        }
        
        predicted_labels, advice = predictor.predict(test_features)
        print(f"\nTest prediction:")
        print(f"Features: {test_features}")
        print(f"Predicted health risks: {predicted_labels}")
        print(f"Advice: {advice}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise


if __name__ == "__main__":
    main()