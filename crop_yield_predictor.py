import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer

yield_df = pd.read_csv(r"E:\crop\yield_df.csv")  

if 'Unnamed: 0' in yield_df.columns:
    yield_df = yield_df.drop(columns=['Unnamed: 0'])

yield_df = yield_df.sample(frac=0.1, random_state=42)  


required_columns = ['temp', 'rainfall', 'pesticides', 'country', 'item', 'year', 'yield']
missing_columns = [col for col in required_columns if col not in yield_df.columns]

if missing_columns:
    print(f"Missing columns in the dataset: {missing_columns}")
else:
    
    X = yield_df[['temp', 'rainfall', 'pesticides', 'country', 'item', 'year']]
    y = yield_df['yield']

    categorical_features = ['country', 'item']
    numerical_features = ['temp', 'rainfall', 'pesticides', 'year']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numerical_features),
            ('cat', OneHotEncoder(), categorical_features)
        ]
    )

    model = make_pipeline(preprocessor, RandomForestRegressor(n_estimators=10, random_state=42))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model.fit(X_train, y_train)

    def predict_yield(country, item, year,temp, rainfall, pesticides):
        input_data = pd.DataFrame({
            'country': [country],
            'item': [item],
            'year': [year],
            'temp': [temp],
            'rainfall': [rainfall],
            'pesticides': [pesticides],
        })
        return model.predict(input_data)[0]

    country = input("Enter country (Like India)): ")
    item = input("Enter item (Maize,Wheat,etc..): ")
    year = int(input("Enter year : "))
    temp = float(input("Enter temperature in Celsius : "))
    rainfall = float(input("Enter rainfall in mm_per_annum: "))
    pesticides = float(input("Enter pesticides in tonnes: "))

    predicted_yield = predict_yield(country, item, year,temp, rainfall, pesticides)
    print(f"Predicted Yield: {predicted_yield}")
