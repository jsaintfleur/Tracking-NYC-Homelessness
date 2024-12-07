import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from colorama import Fore, Style
import importlib
import utilities
importlib.reload(utilities)



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





