import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from colorama import Fore, Style
import importlib
import utilities
importlib.reload(utilities)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from prophet import Prophet


__all__ = [
    "load_data",
    "preprocess_data",
    "summarize_data",
    "plot_time_series",
    "plot_distribution",
    "correlation_analysis",
    "plot_monthly_trends",
    "plot_yearly_trends",
    "generate_report",
    "linear_regression_analysis",
    "forecast_with_prophet",
    "anomaly_detection",
    "generate_machine_learning_report",
]


# Utility Functions
def load_data(file_path):
    """
    Load the dataset from the given file path.
    """
    return pd.read_csv(file_path)

def preprocess_data(df):
    """
    Preprocess the dataset by handling missing values, formatting the date column, etc.
    """
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()
    df['date_of_census'] = pd.to_datetime(df['date_of_census'])
    if df.isnull().values.any():
        print("Missing values detected. Filling with forward fill.")
        df.fillna(method='ffill', inplace=True)
    return df



def summarize_data(df):
    """
    Generate a detailed summary of the dataset, including missing values and descriptive statistics.
    """
    summary = {
        "missing_values": df.isnull().sum(),
        "data_types": df.dtypes,
        "descriptive_stats": df.describe(),
        "unique_dates": df['date_of_census'].nunique(),
        "date_range": (df['date_of_census'].min(), df['date_of_census'].max())
    }
    return summary

def plot_time_series(df, column, title="Time Series Trend"):
    """
    Plot a time series trend for the specified column using Seaborn.
    """
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(14, 8))
    sns.lineplot(data=df, x='date_of_census', y=column, marker="o", color="steelblue", linewidth=2.5)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel(column.replace('_', ' ').title(), fontsize=12)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xticks(rotation=45, fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend([column.replace('_', ' ').title()], loc="upper left", fontsize=12)
    plt.tight_layout()
    plt.show()

def plot_distribution(df, column):
    """
    Plot the distribution of a specific column using Seaborn.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column], kde=True, bins=30)
    plt.title(f"Distribution of {column.replace('_', ' ').title()}")
    plt.xlabel(column.replace('_', ' ').title())
    plt.ylabel("Frequency")
    plt.show()

def plot_monthly_trends(df):
    """
    Plot separate monthly trends for each key variable.
    """
    df['month'] = df['date_of_census'].dt.to_period('M')
    numeric_cols = df.select_dtypes(include='number').columns
    monthly_data = df.groupby('month')[numeric_cols].sum()

    for col in numeric_cols:
        plt.figure(figsize=(12, 6))
        sns.lineplot(x=monthly_data.index.to_timestamp(), y=monthly_data[col], marker="o", label=col.replace('_', ' ').title())
        plt.title(f"Monthly Trends: {col.replace('_', ' ').title()}", fontsize=16, fontweight='bold')
        plt.xlabel("Month", fontsize=12)
        plt.ylabel("Counts", fontsize=12)
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()


def plot_yearly_trends(df):
    """
    Plot separate yearly trends for each key variable.
    """
    df['year'] = df['date_of_census'].dt.year
    numeric_cols = df.select_dtypes(include='number').columns
    yearly_data = df.groupby('year')[numeric_cols].sum()

    for col in numeric_cols:
        plt.figure(figsize=(12, 6))
        sns.lineplot(x=yearly_data.index, y=yearly_data[col], marker="o", label=col.replace('_', ' ').title())
        plt.title(f"Yearly Trends: {col.replace('_', ' ').title()}", fontsize=16, fontweight='bold')
        plt.xlabel("Year", fontsize=12)
        plt.ylabel("Counts", fontsize=12)
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

def generate_correlation_summary(df):
    """
    Generate a user-friendly summary of the correlation matrix with detailed explanations.
    """
    correlation_matrix = df.select_dtypes(include='number').corr()

    print(Fore.YELLOW + "🔗 Correlation Analysis:" + Style.RESET_ALL)
    print("    This section summarizes the relationships between key metrics in the dataset:\n")

    # Total Individuals in Shelter
    total_corr = correlation_matrix.loc['total_individuals_in_shelter']
    print("    - **Total Individuals in Shelter**:")
    print(f"        • Strongly related to Families with Children in Shelter (Correlation: {total_corr['families_with_children_in_shelter']:.2f}).")
    print("          This suggests that families with children are a significant contributor to the total population.")
    print(f"        • Moderately related to Single Adults in Shelter (Correlation: {total_corr['total_single_adults_in_shelter']:.2f}).")
    print("          Single adults also impact the overall shelter numbers.")
    print(f"        • Weakly related to Adult Families in Shelter (Correlation: {total_corr['adult_families_in_shelter']:.2f}).")
    print("          Adult families have a smaller influence on the total count.\n")

    # Single Adults in Shelter
    single_corr = correlation_matrix.loc['total_single_adults_in_shelter']
    print("    - **Single Adults in Shelter**:")
    print(f"        • Strongly correlated with Year (Correlation: {single_corr['year']:.2f}).")
    print("          This reflects a steady increase in single adult shelter usage over time.")
    print(f"        • Weakly related to Families with Children in Shelter (Correlation: {single_corr['families_with_children_in_shelter']:.2f}).")
    print("          Changes in family occupancy have limited impact on single adult trends.\n")

    # Families with Children in Shelter
    families_corr = correlation_matrix.loc['families_with_children_in_shelter']
    print("    - **Families with Children in Shelter**:")
    print(f"        • Very strongly correlated with Total Individuals in Shelter (Correlation: {families_corr['total_individuals_in_shelter']:.2f}).")
    print("          Families with children significantly influence total shelter occupancy.")
    print(f"        • Moderately related to Adult Families in Shelter (Correlation: {families_corr['adult_families_in_shelter']:.2f}).")
    print("          There is a moderate relationship between these groups.\n")

    # Year
    year_corr = correlation_matrix.loc['year']
    print("    - **Year**:")
    print(f"        • Strongly correlated with Single Adults in Shelter (Correlation: {year_corr['total_single_adults_in_shelter']:.2f}).")
    print("          The number of single adults in shelters has grown steadily over the years.")
    print(f"        • Moderately correlated with Total Individuals in Shelter (Correlation: {year_corr['total_individuals_in_shelter']:.2f}).")
    print("          Total shelter usage has also increased over time.")
    print(f"        • Negatively correlated with Adult Families in Shelter (Correlation: {year_corr['adult_families_in_shelter']:.2f}).")
    print("          The number of adult families has not shown a clear trend relative to the years.\n")

    print(Fore.CYAN + "="*60 + Style.RESET_ALL)


def linear_regression_analysis(df, target='total_individuals_in_shelter', correlation_threshold=0.6):
    """
    Perform multi-linear regression to predict the target variable based on high correlating factors.
    
    Parameters:
    - df: DataFrame, the dataset
    - target: str, the target variable to predict
    - correlation_threshold: float, minimum correlation to include a feature
    
    Returns:
    - None
    """
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    
    # Ensure the target variable is in the numeric DataFrame
    if target not in numeric_df.columns:
        print(f"Target variable '{target}' is not numeric or missing from the dataset.")
        return
    
    # Calculate correlations with the target variable
    correlation_matrix = numeric_df.corr()
    high_corr_features = correlation_matrix[target][
        (correlation_matrix[target].abs() >= correlation_threshold) & 
        (correlation_matrix[target].abs() < 1.0)
    ].index.tolist()

    if not high_corr_features:
        print("No features with high correlation found for regression.")
        return
    
    # Define predictors and target
    X = numeric_df[high_corr_features]
    y = numeric_df[target]

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Display results
    print("\nLinear Regression Analysis:")
    print(f"Target Variable: {target.replace('_', ' ').title()}")
    print(f"Features Used: {', '.join([f.replace('_', ' ').title() for f in high_corr_features])}")
    print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
    print(f"R^2 Score: {r2_score(y_test, y_pred):.2f}")
    print("\nCoefficients:")
    for predictor, coef in zip(high_corr_features, model.coef_):
        print(f"  {predictor.replace('_', ' ').title()}: {coef:.2f}")
    print("\n")





def forecast_with_prophet(df):
    """
    Use Prophet to forecast total individuals in shelter.
    """
    forecast_df = df[['date_of_census', 'total_individuals_in_shelter']].rename(
        columns={'date_of_census': 'ds', 'total_individuals_in_shelter': 'y'}
    )
    
    model = Prophet()
    model.fit(forecast_df)

    future = model.make_future_dataframe(periods=365)
    forecast = model.predict(future)

    # Plot forecast
    model.plot(forecast)
    plt.title("Forecast of Total Individuals in Shelter (Prophet)", fontsize=14)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Total Individuals in Shelter", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    # Components plot
    model.plot_components(forecast)
    plt.show()

    print("Prophet Forecast Report:")
    print(f"Forecasted Values for the Next 5 Days:\n{forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head()}\n")

def anomaly_detection(df):
    """
    Perform anomaly detection using Z-Score method.
    """
    df['z_score'] = (df['total_individuals_in_shelter'] - df['total_individuals_in_shelter'].mean()) / \
                    df['total_individuals_in_shelter'].std()
    anomalies = df[df['z_score'].abs() > 3]

    print("Anomaly Detection Report:")
    if anomalies.empty:
        print("No anomalies detected.\n")
    else:
        print(f"Detected {len(anomalies)} anomalies:")
        print(anomalies[['date_of_census', 'total_individuals_in_shelter', 'z_score']])
    print("\n")

def generate_machine_learning_report(df):
    """
    Generate a report combining all machine learning steps.
    """
    print("📊 Machine Learning and Forecasting Report\n")
    print("=" * 60)

    # Linear Regression Analysis
    print("\n🔍 Step 1: Multi-Linear Regression Analysis")
    linear_regression_analysis(df)

    # Prophet Forecasting
    print("\n🔮 Step 2: Time-Series Forecasting with Prophet")
    forecast_with_prophet(df)

    # Anomaly Detection
    print("\n⚠️ Step 3: Anomaly Detection")
    anomaly_detection(df)

    print("=" * 60)
    print("✅ Machine Learning Report Generated Successfully!\n")



