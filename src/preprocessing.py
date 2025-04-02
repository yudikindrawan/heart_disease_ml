import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

def load_and_preprocess_data(input_path, output_path):
  # load dataset
  df = pd.read_csv(input_path)
  
  # Split into X and y
  X = df.drop(columns=['target'])
  y = df['target']
  
  # Label encoding if target kategori
  if y.dtype == 'object':
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)
    joblib.dump(encoder, 'models/label_encoder.pkl')
    
  # Scaling feature
  scaler = StandardScaler()
  X_scaled  = scaler.fit_transform(X)
  joblib.dump(scaler, 'models/scaler.pkl')
  
  # Train-validation-test split
  X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
  X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
  
  # Save the preprocessing results
  pd.DataFrame(X_train).to_csv(output_path + 'X_train.csv', index=False)
  pd.DataFrame(y_train).to_csv(output_path + 'y_train.csv', index=False)
  pd.DataFrame(X_val).to_csv(output_path + 'X_val.csv', index=False)
  pd.DataFrame(y_val).to_csv(output_path + 'y_val.csv', index=False)
  pd.DataFrame(X_test).to_csv(output_path + 'X_test.csv', index=False)
  pd.DataFrame(y_test).to_csv(output_path + 'y_test.csv', index=False)
  
  return X_train, X_val, X_test, y_train, y_val, y_test