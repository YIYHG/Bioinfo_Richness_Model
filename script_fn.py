# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 19:16:25 2024

@author: yiyang huang
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.stats import kruskal
import matplotlib.colors as mcolors


ap = pd.read_excel(r'E:\interview_prep\anu\file_task_PhDVacancyUAntwerp.xlsx') # read the first sheet
md = pd.read_excel(r'E:\interview_prep\anu\file_task_PhDVacancyUAntwerp.xlsx', sheet_name=1) #read the second sheet

ap_cleaned = ap.dropna(subset=['ANPP', 'ANNUAL_TEMPERATURE']) #remove all nan value

############################## Question 1 #####################################


slope, intercept, r_value, p_value, std_err = linregress(ap_cleaned['ANNUAL_TEMPERATURE'], ap_cleaned['ANPP']) # perform linear regression


# print("Slope:", slope)
# print("Intercept:", intercept)
r_squared = r_value ** 2 #r square value
# plot
plt.scatter(ap_cleaned['ANNUAL_TEMPERATURE'], ap_cleaned['ANPP'], label='Data')
plt.plot(ap_cleaned['ANNUAL_TEMPERATURE'], slope * ap_cleaned['ANNUAL_TEMPERATURE'] + intercept, color='red', label=f'ANPP = {slope:.2f} * Temperature + {intercept:.2f}\nR-squared = {r_squared:.2f}')
plt.xlabel('Annual Temperature')
plt.ylabel('ANPP')
plt.title('Linear Model: ANPP vs Annual Temperature')
plt.legend()
plt.show()



correlation_coef = np.corrcoef(ap_cleaned['ANPP'], ap_cleaned['ANNUAL_TEMPERATURE'])[0, 1]
print(f"Correlation coefficient: {correlation_coef:.2f}")



############################## Question 2 #####################################

ap_cleaned['logANPP'] = ap_cleaned['ANPP'].apply(lambda x: np.log(x)) #logANPP

mf = ap_cleaned[ap_cleaned['MANAGEMENT'] == 'M']['logANPP'] # group managed forests
nmf = ap_cleaned[ap_cleaned['MANAGEMENT'] == 'N']['logANPP'] # group not managed forests

statistic, p_value = kruskal(mf, nmf) # kwh test

print(p_value)

############################## Question 3 #####################################


md_GO = md[md['grassland'] == 'GO'] # group go grassland
md_GN = md[md['grassland'] == 'GN'] # group gn grassland

md_GO['total_otu_count'] = md_GO.iloc[:, 3:].sum(axis=1)
sorted_columns = md_GO.drop(['sample_id', 'grassland', 'AverageTemperature', 'total_otu_count'], axis=1).apply(lambda x: x.sort_values(ascending=False).values, axis=1) # rank otu by number

############################## method 1 abundance curve

# plot the abundance curve
calculate_percentage = lambda arr: arr / arr.sum() # calculate relative abundance(%)
percentage_numbers = sorted_columns.apply(calculate_percentage)


# function only keeps data before 0 after ranking
def trim_to_zero(row):
    # Find the index where the value becomes 0
    zero_index = np.where(row == 0)[0]
    if len(zero_index) > 0:
        return row[:zero_index[0] + 1]
    else:
        return row


trimmed_series = percentage_numbers.apply(trim_to_zero) # cutted series


def plot_otu_richness_curve(row):
    
    percentages = row  / row[:200] .sum()*100
    
    plt.plot(range(1, len(row) + 1), percentages, marker='o')

# Plot the OTU richness curve for each row of the Series
plt.figure(figsize=(20, 12))
trimmed_series.apply(plot_otu_richness_curve)
plt.xlabel('OTU numbers', fontsize=14)
plt.ylabel('Relative Abundance (%)', fontsize=14)
plt.title('OTU Richness Curve (GO)', fontsize=16)
plt.grid(True)
plt.show()

# finer figure and plot temp
## plot temperature vs richness

series_40 = trimmed_series.apply(lambda x: x[:40])

temperatures = md_GO['AverageTemperature']  

tr = pd.concat([temperatures, series_40], axis=1) # combine temperature and otu

# Rename the columns
tr.columns = ['Temperature', 'OTU']

# Define a function to plot the OTU richness curve with color based on temperature
def plot_richness_curve_with_color(row):
    percentages = row['OTU'] / row['OTU'].sum() * 100  # Convert to percentages
    cmap = plt.cm.coolwarm  # Choose a colormap from cool to warm
    colors = cmap((row['Temperature'] - tr['Temperature'].min()) / (tr['Temperature'].max() - tr['Temperature'].min()))  # Normalize temperature values and map to colors
    plt.plot(range(1, len(row['OTU']) + 1), percentages, marker='o', color=colors)

# Plot the richness curve for each row with color based on temperature
plt.figure(figsize=(10, 6))
for index, row in tr.iterrows():
    plot_richness_curve_with_color(row)
plt.xlabel('OTU Rank')
plt.ylabel('Relative Abundance (%)')
plt.title('abundance vs Temperature(GO)')
plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=plt.Normalize(vmin=tr['Temperature'].min(), vmax=tr['Temperature'].max())), label='Temperature')
plt.grid(True)
plt.show()


############################## method 2 rarefaction curve
from matplotlib import cm
from matplotlib.colors import Normalize

data = md_GO # can be replaced to md_GN
num_samples_to_pick = range(1, 101)  # From 1 to 100 OTU samples

# visualization 
unique_temperatures = sorted(data['AverageTemperature'].unique()) # exact temp values and setting color
color_map = cm.get_cmap('coolwarm', len(unique_temperatures))
norm = Normalize(vmin=min(unique_temperatures), vmax=max(unique_temperatures))

scalar_mappable = cm.ScalarMappable(cmap=color_map, norm=norm)
scalar_mappable.set_array([])  

legend_handles = []
plt.figure(figsize=(20, 12))


for index, row in data.iterrows(): # iterate over each row 
    
    unique_otus_observed = set() #store observed otus


    cumulative_unique_otus_counts = []

    for num_samples in num_samples_to_pick:
        sampled_otus = row.drop(['sample_id', 'grassland', 'AverageTemperature']).sample(n=num_samples) # randomly select 'num_samples' otus
        
        sampled_values = sampled_otus.values.flatten()
        sampled_values = sampled_values[~pd.isnull(sampled_values)]
        sampled_values = [str(val) for val in sampled_values]
        

        unique_otus_in_sample = set(sampled_values) # count the number of 'new' otus
        unique_otus_observed.update(unique_otus_in_sample) #update unmber
        
        cumulative_unique_otus_counts.append(len(unique_otus_observed)) # append

    temperature_index = unique_temperatures.index(row['AverageTemperature'])
    color = color_map(norm(row['AverageTemperature']))

    plt.plot(num_samples_to_pick, cumulative_unique_otus_counts, marker='o', linestyle='-', color=color)

plt.xlabel('Sample Size')
plt.ylabel('Richness')
plt.gca().yaxis.set_label_position("right")  # y-axis to the right side
plt.gca().yaxis.tick_right() 
plt.title('Richness(GO) vs Temperature')

cbar = plt.colorbar(scalar_mappable, label='Temperature (°C)')


plt.grid(True)
plt.show()

############################## method 3 Shannon index

def shannon_diversity_from_dataframe(dataframe):

    proportions = dataframe.iloc[:, 3:1484].div(dataframe.iloc[:, 3:1484].sum(axis=1), axis=0)
    proportions = proportions.loc[:, (proportions != 0).any(axis=0)]
    

    shannon_indices = -np.sum(proportions * np.log(proportions), axis=1) # calculate index for each row
    
    
    return shannon_indices


data = md_GO #input data, can be replaced by md_GN
shannon_indices = shannon_diversity_from_dataframe(data)


norm = mcolors.Normalize(vmin=data['AverageTemperature'].min(), vmax=data['AverageTemperature'].max())
cmap = plt.get_cmap('coolwarm')

plt.figure(figsize=(8, 6))
sc = plt.scatter(data['AverageTemperature'], shannon_indices, c=data['AverageTemperature'], cmap=cmap, norm=norm, alpha=0.7)
plt.title("Shannon's Diversity Index (GN) vs Average Temperature")
plt.xlabel("Average Temperature")
plt.ylabel("Shannon's Diversity Index")
plt.colorbar(sc, label='Average Temperature')
plt.grid(True)
plt.show()

print(shannon_indices)


