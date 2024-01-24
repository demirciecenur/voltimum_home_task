# Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Function to create and save the dashboard as a PDF
def create_dashboard(file_path, pdf_output_path):
    # Load the data
    data = pd.read_csv(file_path)

    # Setting the style for the plots
    sns.set(style="whitegrid")

    # Creating a figure with multiple subplots
    fig, axes = plt.subplots(2, 1, figsize=(10, 12))

    # Top Topics by Total Events
    top_topics = data.groupby('topic')['events'].sum().sort_values(ascending=False).head(10)
    sns.barplot(x=top_topics.values, y=top_topics.index, ax=axes[0], palette="Blues_d")
    axes[0].set_title('Top 10 Topics by Total Events')
    axes[0].set_xlabel('Total Events')
    axes[0].set_ylabel('Topic')

    # Distribution of Affinity Scores
    sns.histplot(data=data, x='affinity', bins=10, kde=True, ax=axes[1], color="blue")
    axes[1].set_title('Distribution of Affinity Scores')
    axes[1].set_xlabel('Affinity Score')
    axes[1].set_ylabel('Frequency')

    # Adjusting layout
    plt.tight_layout()

    # Save the figure as a PDF
    fig.savefig(pdf_output_path)

# Example usage
csv_file_path = 'output/affinity_score.csv'  # Replace with your CSV file path
pdf_output_path = 'output/dashboard.pdf'  # Replace with your desired output PDF path

# Uncomment the following line to create and save the dashboard
create_dashboard(csv_file_path, pdf_output_path)
