import pandas as pd
import statsmodels.formula.api as smf
import seaborn as sns
import matplotlib.pyplot as plt
import pingouin as pg
import warnings
from matplotlib.backends.backend_pdf import PdfPages
import statsmodels.api as sm

# Column names: ParticipantID,TrialNumber,Response,Correctness,StartTimestamp,EndTimestamp,Immersion,ReactionTime,SampleNumber,SampleOrder,SampleTime,ComparisonTime,Modality,Path,Foil
df = pd.read_csv('data.csv')

# Convert Correctness to error rates (1 for error, 0 for correct)
df['Error'] = df['Correctness'].apply(lambda x: 0 if x else 1)

# Convert relevant columns to categorical types
df['ParticipantID'] = df['ParticipantID'].astype('category')
df['Immersion'] = df['Immersion'].astype('category')
df['Modality'] = df['Modality'].astype('category')

# Aggregate data to get mean error rate and mean response time per participant per condition and rendering type
agg_data = df.groupby(['ParticipantID', 'Immersion', 'Modality'], observed=False).agg({'Error': 'mean', 'ReactionTime': 'mean'}).reset_index()

# Remove rows with NaN values in the Error column
agg_data = agg_data.dropna(subset=['Error', 'ReactionTime'])

# Check the cleaned aggregated data
print("\nCleaned aggregated data:")
print(agg_data.head())

# Perform the Shapiro-Wilk test for the 'Error' column
normality_test_error = pg.normality(agg_data['Error'], method='shapiro')
print(normality_test_error)

# Perform the Shapiro-Wilk test for the 'ReactionTime' column
normality_test_reaction_time = pg.normality(agg_data['ReactionTime'], method='shapiro')
print(normality_test_reaction_time)

# Fit a Linear Mixed Model for Error Rates with specified optimizers
error_model = smf.mixedlm("Error ~ Immersion * Modality", agg_data, groups=agg_data["ParticipantID"], re_formula="~Modality")
error_fit = error_model.fit(method='nm', maxiter=1000, full_output=True)  # Nelder-Mead
if not error_fit.converged:
    error_fit = error_model.fit(method='bfgs', maxiter=1000, full_output=True)  # BFGS
print(error_fit.summary())

# Fit a Linear Mixed Model for Reaction Times with specified optimizers
reaction_time_model = smf.mixedlm("ReactionTime ~ Immersion * Modality", agg_data, groups=agg_data["ParticipantID"], re_formula="~Modality")
reaction_time_fit = reaction_time_model.fit(method='nm', maxiter=1000, full_output=True)  # Nelder-Mead
if not reaction_time_fit.converged:
    reaction_time_fit = reaction_time_model.fit(method='bfgs', maxiter=1000, full_output=True)  # BFGS
print(reaction_time_fit.summary())


# Disabling excessive warnings from plotting
warnings.filterwarnings('ignore')

# Diagnostics: Residual plots and random effects

# Residual plot for error model

plt.rcParams['pdf.fonttype'] = 42

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
sm.qqplot(error_fit.resid, line='s', ax=ax[0])
ax[0].set_title('Q-Q plot of residuals for error model')
ax[0].set_xlabel('Theoretical Quantiles')
ax[0].set_ylabel('Standardized Residuals')

# Residual plot for reaction time model
sm.qqplot(reaction_time_fit.resid, line='s', ax=ax[1])
ax[1].set_title('Q-Q plot of residuals for reaction time model')
ax[1].set_xlabel('Theoretical Quantiles')
ax[1].set_ylabel('Standardized Residuals')
plt.tight_layout()
plt.show()

# Random effects for error model
print("\nRandom effects for error model:")
print(error_fit.random_effects)

# Random effects for reaction time model
print("\nRandom effects for reaction time model:")
print(reaction_time_fit.random_effects)

# Create a new PDF file for plots
pdf_pages = PdfPages('interaction_plots.pdf')

# Create a new figure with two subplots side by side
fig, axs = plt.subplots(1, 2, figsize=(16, 7))


# Visualization of the interaction effects on Error Rates
sns.pointplot(x='Modality', y='Error', hue='Immersion', data=agg_data, dodge=True, markers=['o', 's'], capsize=0.1, errwidth=1, palette='colorblind', ax=axs[0])
axs[0].set_title('Interaction of Display Environment and Sensory Modality on Error Rate', fontsize=13, pad=10)
axs[0].set_xlabel('Sensory Modality', labelpad=20, fontsize=13)
axs[0].set_ylabel('Error Rate', labelpad=10, fontsize=12)
# axs[0].legend(title='Display Environment')
handles, labels = axs[0].get_legend_handles_labels()
axs[0].legend(handles=handles, labels=['Projected Surface', 'Virtual Reality'], title='Display Environment', fontsize=12, title_fontsize=12)
axs[0].set_xticklabels(['Haptic', 'Visual', 'Visuohaptic'])

offset = 0.03


# Add significance annotations for Error Rates
# Visual
axs[0].text(1 + offset, 0.303, '*', ha='center', fontsize=15)
# Visuohaptic
axs[0].text(2 + offset, 0.247, '*', ha='center', fontsize=15)

plt.sca(axs[0])
plt.xticks(fontsize=13)
plt.yticks(fontsize=12)


# Visualization of the interaction effects on Reaction Times
sns.pointplot(x='Modality', y='ReactionTime', hue='Immersion', data=agg_data, dodge=True, markers=['o', 's'], capsize=0.1, errwidth=1, palette='colorblind', ax=axs[1])
axs[1].set_title('Interaction of Display Environment and Sensory Modality on Response Time', fontsize=13, pad=10)
axs[1].set_xlabel('Sensory Modality', labelpad=20, fontsize=13)
axs[1].set_ylabel('Response Time', labelpad=10, fontsize=12)
# axs[1].legend(title='Display Environment')
axs[1].legend(handles=handles, labels=['Projected Surface', 'Virtual Reality'], title='Display Environment', fontsize=12, title_fontsize=12)
axs[1].set_xticklabels(['Haptic', 'Visual', 'Visuohaptic'])

axs[1].set_ylim(6.4, 12.4)

# Add significance annotations for Reaction Times 
# Haptic
axs[1].text(0 + offset, 12.075, '*', ha='center', fontsize=15)
# Visual
axs[1].text(1 + offset, 10.815, '*', ha='center', fontsize=15)
# Visuohaptic
axs[1].text(2 + offset, 10.71, '*', ha='center', fontsize=15)

plt.sca(axs[1])

plt.xticks(fontsize=13)
plt.yticks(fontsize=12)


# Adjust the layout of the figure
plt.tight_layout()

plt.show()

# Save the current figure to the PDF file
pdf_pages.savefig(fig)

# Close the PDF file
pdf_pages.close()

# Calculate mean error rate and mean response time per condition per rendering
mean_data = agg_data.groupby(['Immersion', 'Modality']).agg({'Error': 'mean', 'ReactionTime': 'mean'}).reset_index()

# Print mean error rate and mean response time per condition per rendering
print("\nMean Error Rate and Mean Response Time per Modality per Environment:")
print(mean_data)

# Generate Paragraphs for the Results Section
for idx, row in mean_data.iterrows():
    print(f"\nFor {row['Immersion']} rendering:")
    print(f"In the {row['Modality']} condition:")
    print(f"- Mean error rate: {row['Error']:.2f}")
    print(f"- Mean response time: {row['ReactionTime']:.2f} seconds")
    
