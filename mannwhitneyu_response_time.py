from scipy.stats import mannwhitneyu
import pandas as pd

# Load the data from the uploaded CSV file
file_path = 'data.csv'
data = pd.read_csv(file_path)

# Function to perform Mann-Whitney U test
def mann_whitney_test(data, modality):
    # Filter data by modality
    modality_data = data[data['Modality'] == modality]
    
    # Separate data into VR and Desktop groups
    vr_group = modality_data[modality_data['Immersion'] == 'VR']['ReactionTime']
    desktop_group = modality_data[modality_data['Immersion'] == 'Desktop']['ReactionTime']
    
    # Perform Mann-Whitney U test
    stat, p_value = mannwhitneyu(vr_group, desktop_group, alternative='two-sided')
    
    return stat, p_value

# Perform the test for each modality
results = {}
modalities = ['H', 'V', 'VH']
for modality in modalities:
    results[modality] = mann_whitney_test(data, modality)

# Display the results in a formatted DataFrame
results_df = pd.DataFrame(results, index=['Statistic', 'p-value']).T
results_df['Statistic'] = results_df['Statistic'].apply(lambda x: f"{x:.2f}")
results_df['p-value'] = results_df['p-value'].apply(lambda x: "<0.001" if x < 0.001 else f"{x:.3f}")

print(results_df)