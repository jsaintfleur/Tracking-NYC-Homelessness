from utilities import (
    load_data,
    preprocess_data,
    summarize_data,
    plot_time_series,
    correlation_analysis,
    plot_distribution,
    monthly_trends,
    yearly_trends,
    generate_report
)

def main():
    """
    Main function to orchestrate the workflow using utility functions.
    """
    # Path to the dataset
    file_path = 'DHS_Homeless_Shelter_Census.csv'
    
    # Load and preprocess data
    print("Loading data...")
    df = load_data(file_path)
    print("Preprocessing data...")
    df = preprocess_data(df)
    
    # Summarize data
    print("\nDataset Summary:")
    summary = summarize_data(df)
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    # Plot time series trends
    print("\nPlotting time series for Total Individuals in Shelter...")
    plot_time_series(df, 'total_individuals_in_shelter', "Total Individuals in Shelter Over Time")
    
    # Analyze correlations
    print("\nPerforming correlation analysis...")
    correlation_analysis(df)
    
    # Plot distribution of a column
    print("\nPlotting distribution for Total Single Adults in Shelter...")
    plot_distribution(df, 'total_single_adults_in_shelter')
    
    # Generate monthly and yearly trends
    print("\nGenerating monthly trends...")
    monthly_trends(df)
    print("\nGenerating yearly trends...")
    yearly_trends(df)
    
    # Generate a comprehensive console report
    print("\nGenerating report...")
    generate_report(df)

if __name__ == "__main__":
    main()
