import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np


df = pd.read_csv("veriseti_son_hal.csv")

X = df.drop('puan', axis=1)
y = df['puan']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

numerical_cols = ['sezon', 'yarışmacı kodu', 'yükseklik', 'sayı değeri', 'yol faktörü', 'rüzgar faktörü']

scaler = StandardScaler()
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

for col in X_train.columns:
    if X_train[col].dtype == 'bool':
        X_train[col] = X_train[col].astype(int)
    if X_test[col].dtype == 'bool':
        X_test[col] = X_test[col].astype(int)


models = {
    'Çoklu Doğrusal Regresyon': {
        'model': LinearRegression(),
        'params': {} 
    },
    'SVR': {
        'model': SVR(),
        'params': {
            'kernel': ['rbf'],
            'C': [0.1, 1, 10],
            'epsilon': [0.1, 0.01]
        }
    },
    'RandomForestRegressor': {
        'model': RandomForestRegressor(random_state=42),
        'params': {
            'n_estimators': [50, 100],
            'max_depth': [10, 20],
            'min_samples_split': [2, 5]
        }
    },
    'GradientBoostingRegressor (GBM)': {
        'model': GradientBoostingRegressor(random_state=42),
        'params': {
            'n_estimators': [50, 100],
            'learning_rate': [0.1, 0.01],
            'max_depth': [3, 5]
        }
    }
}

results = []


for name, setup in models.items():
    print(f"##  {name} İşlemi Başlatılıyor...")

    if setup['params']:
        grid_search = GridSearchCV(
            estimator=setup['model'],
            param_grid=setup['params'],
            scoring='neg_mean_squared_error',
            cv=3,
            n_jobs=-1,
            verbose=0 
        )
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
    
    else:
        best_model = setup['model']
        best_model.fit(X_train, y_train)
        best_params = 'N/A'

    y_pred = best_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    results.append({
        'Model Adı': name,
        'RMSE': rmse,
        'R²': r2,
        'En İyi Parametreler': best_params
    })

    print(f"\n###  {name} Sonuçları")
    print(f"- **En İyi Hiperparametreler:** {best_params}")
    print(f"- **Test RMSE:** {rmse:.4f}")
    print(f"- **Test R²:** {r2:.4f}")
    print("-" * 50)



results_df = pd.DataFrame(results)
results_df_summary = results_df[['Model Adı', 'RMSE', 'R²']].sort_values(by='RMSE')

print("\n\n##  Modellerin Test Performansı Karşılaştırma Tablosu")
print(results_df_summary.to_markdown(index=False, floatfmt=".4f"))
print("\n" + "=" * 50 + "\n")

best_model_row = results_df_summary.iloc[0]
best_model_name = best_model_row['Model Adı']
best_rmse = best_model_row['RMSE']
best_r2 = best_model_row['R²']

print("##  Genel Değerlendirme")
print(f"En başarılı model **{best_model_name}** olarak belirlenmiştir.")
print(f"Bu model, **{best_r2:.4f}** ile en yüksek **R²** (Belirlilik Katsayısı) değerini ve **{best_rmse:.4f}** ile en düşük **RMSE** (Hata) değerini göstermiştir.")

if 'GradientBoostingRegressor' in best_model_name or 'RandomForestRegressor' in best_model_name:
    print("Ensemble (topluluk) modellerinin (GBM ve Random Forest) bu kadar başarılı olması, puanlama üzerindeki faktörler arasında **karmaşık ve doğrusal olmayan** ilişkiler olduğunu işaret eder. Bu modeller, verideki bu karmaşıklığı basit Doğrusal Regresyondan daha iyi yakalamıştır.")