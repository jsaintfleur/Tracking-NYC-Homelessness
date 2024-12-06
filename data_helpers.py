import pandas as pd
import matplotlib.pyplot as plt

def load_data(file_path):
    """
    Load the dataset from the given file path.
    """
    return pd.read_csv(file_path)

def preprocess_data(df):
    """
    Preprocess the dataset by handling missing values, formatting date column, etc.
    """
    # Convert the Date column to datetime
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    return df

def summarize_data(df):
    """
    Generate a summary of the dataset, including missing values and descriptive statistics.
    """
    summary = {
        "missing_values": df.isnull().sum(),
        "data_types": df.dtypes,
        "descriptive_stats": df.describe()
    }
    return summary

def plot_trend(df, column, title="Trend Over Time"):
    """
    Plot the trend of a specific column over time.
    """
    if 'Date' not in df.columns or column not in df.columns:
        raise ValueError("The DataFrame must have a 'Date' column and the specified column.")
    
    plt.figure(figsize=(10, 6))
    plt.plot(df['Date'], df[column], label=column)
    plt.xlabel("Date")
    plt.ylabel(column)
    plt.title(title)
    plt.legend()
    plt.show()

def daily_report(df, date):
    """
    Generate a daily report for the given date.
    """
    if 'Date' not in df.columns:
        raise ValueError("The DataFrame must have a 'Date' column.")
    
    # Filter data for the given date
    report = df[df['Date'] == pd.to_datetime(date)]
    
    if report.empty:
        return f"No data available for {date}."
    
    return report
