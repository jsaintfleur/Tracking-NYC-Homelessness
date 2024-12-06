from helpers.data_helpers import load_and_preprocess_data, summarize_data, generate_insights, plot_trends, correlation_analysis

# File path to the dataset
file_path = "data/DHS_Homeless_Shelter_Census.csv"

# Step 1: Load and preprocess the data
data = load_and_preprocess_data(file_path)

# Step 2: Summarize the data
summarize_data(data)

# Step 3: Generate high-level insights
generate_insights(data)

# Step 4: Plot trends for different categories
plot_trends(data)

# Step 5: Perform correlation analysis
correlation_analysis(data)
