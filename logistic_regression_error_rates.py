import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
file_path = 'data.csv'
data = pd.read_csv(file_path)

# Convert Correctness column to numeric
data['Correctness'] = data['Correctness'].astype(int)

# Define the modalities to analyze
modalities = ['H', 'V', 'VH']

# Calculate the error rate for each trial
data['ErrorRate'] = data['Correctness'].apply(lambda x: 0 if x else 1)

# Fit a logistic regression model for each modality
logit_results = {}
for modality in modalities:
    # Filter data by modality
    modality_data = data[data['Modality'] == modality]
    
    # Fit the logistic regression model
    model = smf.logit("Correctness ~ Immersion", data=modality_data)
    result = model.fit()
    logit_results[modality] = result.summary2().tables[1]

# Combine the logistic regression results into one DataFrame for easier comparison
logit_summary_df = pd.concat(logit_results, axis=0, keys=logit_results.keys())

# Format the results for reporting
logit_summary_df = logit_summary_df.reset_index()
logit_summary_df['Coef.'] = logit_summary_df['Coef.'].apply(lambda x: f"{x:.2f}")
logit_summary_df['Std.Err.'] = logit_summary_df['Std.Err.'].apply(lambda x: f"{x:.2f}")
logit_summary_df['z'] = logit_summary_df['z'].apply(lambda x: f"{x:.2f}")
logit_summary_df['P>|z|'] = logit_summary_df['P>|z|'].apply(lambda x: "<0.001" if x < 0.001 else f"{x:.3f}")
logit_summary_df['[0.025'] = logit_summary_df['[0.025'].apply(lambda x: f"{x:.2f}")
logit_summary_df['0.975]'] = logit_summary_df['0.975]'].apply(lambda x: f"{x:.2f}")

# Print the formatted logistic regression summary
print(logit_summary_df)