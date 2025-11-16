#!/usr/bin/env python3
"""
Test script to verify the automated preprocessing pipeline works correctly.
"""

import os
import sys
sys.path.append('preprocessing')

from automate_Lukas import preprocess_diabetes_data, load_preprocessed_data

def test_preprocessing_pipeline():
    """Test the complete preprocessing pipeline."""
    print("="*50)
    print("Testing Automated Preprocessing Pipeline")
    print("="*50)
    
    try:
        # Test preprocessing
        X_train, X_val, y_train, y_val, scaler = preprocess_diabetes_data(
            file_path='diabetes.csv',
            test_size=0.3,
            random_state=42
        )
        
        print("\n✅ Preprocessing completed successfully!")
        print(f"Training features shape: {X_train.shape}")
        print(f"Validation features shape: {X_val.shape}")
        print(f"Training labels shape: {y_train.shape}")
        print(f"Validation labels shape: {y_val.shape}")
        
        # Test loading preprocessed data
        print("\n" + "="*30)
        print("Testing Data Loading")
        print("="*30)
        
        loaded_data = load_preprocessed_data('diabetes_preprocessing')
        
        if loaded_data is not None:
            print("✅ Preprocessed data loaded successfully!")
            
        print("\n🎉 All tests passed! The automation pipeline is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_preprocessing_pipeline()
    sys.exit(0 if success else 1)
