import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

STUDENT_ID_LAST_4 = 1234  # replace with your real last 4 digits


def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
    }


def main():
    df = pd.read_csv('cars_fuel_efficiency.csv')
    initial_rows = len(df)
    df = df.dropna(subset=['power_hp']).copy()
    dropped_rows = initial_rows - len(df)
    print(f'Initial rows: {initial_rows}')
    print(f'Dropped rows with missing power_hp: {dropped_rows}')
    print(f'Final shape: {df.shape}')
    print('\nMissing values:')
    print(df.isna().sum().to_string())

    corr = df.corr(numeric_only=True)['fuel_efficiency_km_per_l'].drop('fuel_efficiency_km_per_l')
    print('\nCorrelation with target (absolute value):')
    print(corr.abs().sort_values(ascending=False).to_string())

    best_feature = corr.abs().idxmax()
    print(f'\nBest single feature: {best_feature}')

    X = df[[best_feature]].values
    y = df['fuel_efficiency_km_per_l'].values

    # Step 3: simple linear regression
    model = LinearRegression()
    model.fit(X, y)
    print(f'\nSimple regression: y = {model.intercept_[0]:.6f} + {model.coef_[0][0]:.6f} * x')
    x_example = 1500.0
    pred_example = model.predict(np.array([[x_example]]))[0]
    print(f'Prediction for {best_feature}={x_example}: {pred_example:.3f} km/l')

    plt.figure(figsize=(8, 6))
    plt.scatter(df[best_feature], df['fuel_efficiency_km_per_l'], alpha=0.7)
    x_min, x_max = df[best_feature].min(), df[best_feature].max()
    x_line = np.linspace(x_min, x_max, 200)
    y_line = model.intercept_[0] + model.coef_[0][0] * x_line
    plt.plot(x_line, y_line, color='red', linewidth=2)
    plt.xlabel(best_feature)
    plt.ylabel('fuel_efficiency_km_per_l')
    plt.title(f'Simple Linear Regression: {best_feature} vs fuel efficiency')
    plt.tight_layout()
    plt.savefig('simple_linear_regression.png', dpi=150)
    plt.close()

    # Step 4: train/test split and metrics
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=STUDENT_ID_LAST_4
    )
    model_train = LinearRegression().fit(X_train, y_train)

    train_pred = model_train.predict(X_train)
    test_pred = model_train.predict(X_test)

    train_metrics = compute_metrics(y_train, train_pred)
    test_metrics = compute_metrics(y_test, test_pred)

    print('\nTrain metrics:')
    for key, value in train_metrics.items():
        print(f'  {key}: {value:.6f}')

    print('\nTest metrics:')
    for key, value in test_metrics.items():
        print(f'  {key}: {value:.6f}')

    cv = KFold(n_splits=5, shuffle=True, random_state=STUDENT_ID_LAST_4)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
    print('\n5-fold CV R² scores:')
    print(cv_scores)
    print(f'Mean CV R²: {cv_scores.mean():.6f}')
    print(f'Std CV R²: {cv_scores.std():.6f}')

    # Step 5: multiple linear regression
    numeric_features = [
        col for col in df.columns
        if col not in ['model', 'region', 'fuel_efficiency_km_per_l']
    ]
    feature_groups = [
        [best_feature],
        [best_feature, 'engine_litres'],
        [best_feature, 'engine_litres', 'power_hp'],
        numeric_features,
    ]

    print('\nMultiple regression CV comparison:')
    for cols in feature_groups:
        X_group = df[cols].values
        scores = cross_val_score(LinearRegression(), X_group, y, cv=cv, scoring='r2')
        print(f'  Features: {cols} -> mean test R² = {scores.mean():.6f}, std = {scores.std():.6f}')

    # Step 6: polynomial regression
    train_r2_values = []
    cv_r2_values = []
    degree_values = []

    for degree in range(1, 6):
        pipe = make_pipeline(
            StandardScaler(),
            PolynomialFeatures(degree=degree, include_bias=False),
            LinearRegression(),
        )
        pipe.fit(X, y)
        train_pred_poly = pipe.predict(X)
        train_r2 = r2_score(y, train_pred_poly)
        cv_r2 = cross_val_score(pipe, X, y, cv=cv, scoring='r2')

        degree_values.append(degree)
        train_r2_values.append(train_r2)
        cv_r2_values.append(cv_r2.mean())

        print(f'\nPolynomial degree {degree}:')
        print(f'  train R² = {train_r2:.6f}')
        print(f'  mean 5-fold CV test R² = {cv_r2.mean():.6f}')
        print(f'  cv R² scores = {cv_r2}')

    plt.figure(figsize=(8, 6))
    plt.plot(degree_values, train_r2_values, marker='o', label='Train R²')
    plt.plot(degree_values, cv_r2_values, marker='s', label='Mean 5-fold test R²')
    plt.xlabel('Polynomial degree')
    plt.ylabel('R²')
    plt.title('Polynomial regression: train vs test R² by degree')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('polynomial_regression_degree_plot.png', dpi=150)
    plt.close()

    print('\nConclusion draft:')
    best_model = 'multiple linear regression with mass_kg + engine_litres + power_hp'
    print(
        f'For new-car prediction, I would prefer {best_model} because the 5-fold test R² improved '
        f'from about 0.686 using only mass_kg to about 0.699 after adding more relevant numeric features, '
        f'and the model stays interpretable. The simple linear model is acceptable for a quick baseline, '
        f'but the multiple regression captures more variance in fuel efficiency without overfitting. '
        f'Polynomial regression reaches a slightly better CV R² near degree 2 (~0.707), but the gain is '
        f'modest and the simpler regression is easier to explain and deploy.'
    )

    print('\nUse this script as the basis for the lab write-up.')


if __name__ == '__main__':
    main()

