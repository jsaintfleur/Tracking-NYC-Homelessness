import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    """
    Load the dataset from the given file path.
    """
    return pd.read_csv(file_path)

def preprocess_data(df):
    """
    Preprocess the dataset by handling missing values, formatting the date column, etc.
    """
    # Rename columns for easier handling
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()
    
    # Convert the 'date_of_census' column to datetime
    df['date_of_census'] = pd.to_datetime(df['date_of_census'])
    
    # Check for missing values and fill or remove them
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
    Plot a time series trend for the specified column.
    """
    if 'date_of_census' not in df.columns or column not in df.columns:
        raise ValueError("The DataFrame must have 'date_of_census' and the specified column.")
    
    plt.figure(figsize=(12, 6))
    plt.plot(df['date_of_census'], df[column], label=column, marker='o')
    plt.xlabel("Date")
    plt.ylabel(column.replace('_', ' ').title())
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()

def daily_report(df, date):
    """
    Generate a daily report for the given date.
    """
    if 'date_of_census' not in df.columns:
        raise ValueError("The DataFrame must have a 'date_of_census' column.")
    
    date = pd.to_datetime(date)
    report = df[df['date_of_census'] == date]
    
    if report.empty:
        return f"No data available for {date.strftime('%Y-%m-%d')}."
    return report

def correlation_analysis(df):
    """
    Analyze correlations between numerical columns.
    """
    correlation = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm')
    plt.title("Correlation Matrix")
    plt.show()
    return correlation

def plot_distribution(df, column):
    """
    Plot the distribution of a specific column.
    """
    if column not in df.columns:
        raise ValueError(f"The specified column '{column}' is not in the DataFrame.")
    
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column], kde=True, bins=30)
    plt.title(f"Distribution of {column.replace('_', ' ').title()}")
    plt.xlabel(column.replace('_', ' ').title())
    plt.ylabel("Frequency")
    plt.show()

def monthly_trends(df):
    """
    Aggregate data by month and visualize monthly trends.
    """
    df['month'] = df['date_of_census'].dt.to_period('M')
    monthly_data = df.groupby('month').sum()
    
    monthly_data.plot(figsize=(12, 6), marker='o')
    plt.title("Monthly Trends")
    plt.ylabel("Counts")
    plt.xlabel("Month")
    plt.grid(True)
    plt.show()
    
    return monthly_data

def yearly_trends(df):
    """
    Aggregate data by year and visualize yearly trends.
    """
    df['year'] = df['date_of_census'].dt.year
    yearly_data = df.groupby('year').sum()
    
    yearly_data.plot(figsize=(12, 6), marker='o')
    plt.title("Yearly Trends")
    plt.ylabel("Counts")
    plt.xlabel("Year")
    plt.grid(True)
    plt.show()
    
    return yearly_data

def generate_report(df):
    """
    Generate a comprehensive console report summarizing the data.
    """
    # Key statistics
    total_individuals_avg = df['total_individuals_in_shelter'].mean()
    single_adults_avg = df['total_single_adults_in_shelter'].mean()
    families_with_children_avg = df['families_with_children_in_shelter'].mean()
    adult_families_avg = df['adult_families_in_shelter'].mean()
    
    # Date range
    start_date = df['date_of_census'].min()
    end_date = df['date_of_census'].max()
    
    # Trends
    print("\nHomeless Shelter Census Report")
    print("="*50)
    print(f"Date Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Total Entries: {len(df)}")
    print("\nKey Averages:")
    print(f" - Average Total Individuals in Shelter: {total_individuals_avg:.2f}")
    print(f" - Average Single Adults in Shelter: {single_adults_avg:.2f}")
    print(f" - Average Families with Children in Shelter: {families_with_children_avg:.2f}")
    print(f" - Average Adult Families in Shelter: {adult_families_avg:.2f}")
    
    print("\nMonthly Trends (Sum of Individuals in Shelter):")
    monthly_data = df.groupby(df['date_of_census'].dt.to_period('M')).sum()
    print(monthly_data['total_individuals_in_shelter'].head(12))  # Print the first 12 months
    
    print("\nYearly Trends (Sum of Individuals in Shelter):")
    yearly_data = df.groupby(df['date_of_census'].dt.year).sum()
    print(yearly_data['total_individuals_in_shelter'])
    
    print("\nDetailed Correlation Analysis:")
    print(df.corr())

if __name__ == "__main__":
    # File Usage
    file_path = 'DHS_Homeless_Shelter_Census.csv'
    df = load_data(file_path)
    df = preprocess_data(df)

    # Generate summary and visualizations
    summary = summarize_data(df)
    print(summary)

    plot_time_series(df, 'total_individuals_in_shelter', "Total Individuals in Shelter Over Time")
    correlation = correlation_analysis(df)
    plot_distribution(df, 'total_single_adults_in_shelter')
    monthly_data = monthly_trends(df)
    yearly_data = yearly_trends(df)

    # Generate report in console
    generate_report(df)

