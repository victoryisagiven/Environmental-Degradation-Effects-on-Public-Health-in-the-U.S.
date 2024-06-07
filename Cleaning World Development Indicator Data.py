# -*- coding: utf-8 -*-
"""
Created on Sat May 25 16:20:44 2024

@author: books
"""
#import libraries
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
import seaborn as sns

#read in the data
env = pd.read_csv('C:\\Users\\books\\OneDrive\\Documents\\DS Projects\\Environmental Abuses on Human Health\\WDICSV.csv')
env.info()

#drop all the rows except for the united states
condition = (env['Country Name'] != 'United States')
us = env.drop(env[condition].index)
us.info()

#separate data by numeric and categorical data
us1 = us.drop(['1960', '1961', '1962', '1963', '1964', '1965', '1966','1967',
               '1968', '1969', '1970', '1971', '1972', '1973', '1974', '1975', 
               '1976', '1977', '1978', '1979', '1980', '1981', '1982', '1983', 
               '1984', '1985', '1986', '1987', '1988', '1989', '1990', '1991', 
               '1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999', 
               '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', 
               '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', 
               '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023'], axis=1)

us1.info()
us1.head()
us1 = us1.reset_index(drop=True)

us2 = us.drop(['Country Name', 'Country Code', 'Indicator Name', 
               'Indicator Code'], axis = 1)

#impute missing values in numeric data
imputer = KNNImputer(n_neighbors=5)
us2 = imputer.fit_transform(us2)
us2 = pd.DataFrame(us2)

us2.info()
us2.tail()

#rename the columns by the year
us2.columns = ['1960', '1961', '1962', '1963', '1964', '1965', '1966','1967',
               '1968', '1969', '1970', '1971', '1972', '1973', '1974', '1975', 
               '1976', '1977', '1978', '1979', '1980', '1981', '1982', '1983', 
               '1984', '1985', '1986', '1987', '1988', '1989', '1990', '1991', 
               '1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999', 
               '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', 
               '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', 
               '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023']

#rejoin the categorical and numerical data
usdf = pd.concat([us1, us2], axis=1)

usdf.info()
usdf.tail()


#export the file to only get the rows needed
from pathlib import Path  
filepath = Path('C:\\Users\\books\\OneDrive\\Documents\\DS Projects\\Environmental Abuses on Human Health\\CleanedUSData.csv')  
filepath.parent.mkdir(parents=True, exist_ok=True)  
usdf.to_csv(filepath) 

#import the file once again
us = pd.read_csv('C:\\Users\\books\\OneDrive\\Documents\\DS Projects\\Environmental Abuses on Human Health\\CleanedUSData2.csv')
us.info()

#delete unnecessary columns
usdf = us.drop(['Unnamed: 0', 'Country Name', 'Country Code', 'Indicator Code'], axis=1)
usdf.info()

#Set index to Indicator column
us1 = usdf.set_index('Indicator Name')
us1.head()

#Transpose the data
usdf = us1.T
usdf.head()
usdf.info()

#rename the columns
usdf.columns
usdf.columns = ['Carbon dioxide damage', 'Energy depletion', 'Mineral depletion',
       'Natural resources depletion', 'Net forest depletion', 'Particulate emission damage',
       'Annual freshwater withdrawals', 'CO2 emissions', 'Death rate',
       'Diabetes prevalence', 'Fertility rate', 'Water stress', 'Life expectancy',
       'Methane emissions', 'Mortality from CVD, cancer, diabetes or CRD',
       'Mortality rate attributed to air pollution', 'Adult female mortality rate',
       'Adult male mortality rate', 'Infant mortality rate', 'Nitrous oxide emissions',
       'Air pollution', 'Female Survival to age 65', 'Male Survival to age 65',
       'Greenhouse gas emissions']


#export the cleaned data
from pathlib import Path  
filepath = Path('C:\\Users\\books\\OneDrive\\Documents\\DS Projects\\Environmental Abuses on Human Health\\FinalCleanedUSData.csv')  
filepath.parent.mkdir(parents=True, exist_ok=True)  
usdf.to_csv(filepath) 


#Calculate the correlation matrix
correlation_matrix = usdf.corr()

#show heatmap of the correlations
sns.heatmap(correlation_matrix, 
            xticklabels=correlation_matrix.columns.values,
            yticklabels=correlation_matrix.columns.values)

#get the top correlations
def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, n=5):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]

top_corr = get_top_abs_correlations(usdf, 200)
top_corr.info()
top_corr = top_corr.to_frame()

#ecport the correlation data
from pathlib import Path  
filepath = Path('C:\\Users\\books\\OneDrive\\Documents\\DS Projects\\Environmental Abuses on Human Health\\Top Correlations.csv')  
filepath.parent.mkdir(parents=True, exist_ok=True)  
top_corr.to_csv(filepath)

from scipy.stats import pearsonr

#we'll see if the relevant correlation coefficients are statistically significant
#co2 damage and female survival
r, p_value = pearsonr(usdf['Carbon dioxide damage'],usdf['Female Survival to age 65'] )

# Print results
print(f"Pearson correlation coefficient: {r}")
print(f"p-value: {p_value}")

# Check if p-value is less than the significance level (e.g., 0.05)
if p_value < 0.05:
    print("The correlation is statistically significant.")
else:
    print("The correlation is not statistically significant.")

#co2 damage and life expectancy
r, p_value = pearsonr(usdf['Carbon dioxide damage'],usdf['Life expectancy'] )

# Print results
print(f"Pearson correlation coefficient: {r}")
print(f"p-value: {p_value}")

# Check if p-value is less than the significance level (e.g., 0.05)
if p_value < 0.05:
    print("The correlation is statistically significant.")
else:
    print("The correlation is not statistically significant.")

#Freshwater withdrawals and male mortality
r, p_value = pearsonr(usdf['Annual freshwater withdrawals'],usdf['Adult male mortality rate'] )

# Print results
print(f"Pearson correlation coefficient: {r}")
print(f"p-value: {p_value}")

# Check if p-value is less than the significance level (e.g., 0.05)
if p_value < 0.05:
    print("The correlation is statistically significant.")
else:
    print("The correlation is not statistically significant.")
    
#methane emissions and male survival
r, p_value = pearsonr(usdf['Methane emissions'],usdf['Male Survival to age 65'] )

# Print results
print(f"Pearson correlation coefficient: {r}")
print(f"p-value: {p_value}")

# Check if p-value is less than the significance level (e.g., 0.05)
if p_value < 0.05:
    print("The correlation is statistically significant.")
else:
    print("The correlation is not statistically significant.")
    
#methane emission and female survival
r, p_value = pearsonr(usdf['Methane emissions'],usdf['Female Survival to age 65'] )

# Print results
print(f"Pearson correlation coefficient: {r}")
print(f"p-value: {p_value}")

# Check if p-value is less than the significance level (e.g., 0.05)
if p_value < 0.05:
    print("The correlation is statistically significant.")
else:
    print("The correlation is not statistically significant.")

#mortality from diseases and air pollution
r, p_value = pearsonr(usdf['Mortality from CVD, cancer, diabetes or CRD'],usdf['Air pollution'] )

# Print results
print(f"Pearson correlation coefficient: {r}")
print(f"p-value: {p_value}")

# Check if p-value is less than the significance level (e.g., 0.05)
if p_value < 0.05:
    print("The correlation is statistically significant.")
else:
    print("The correlation is not statistically significant.")
    
#life expectancy and methane emissions
r, p_value = pearsonr(usdf['Life expectancy'],usdf['Methane emissions'] )

# Print results
print(f"Pearson correlation coefficient: {r}")
print(f"p-value: {p_value}")

# Check if p-value is less than the significance level (e.g., 0.05)
if p_value < 0.05:
    print("The correlation is statistically significant.")
else:
    print("The correlation is not statistically significant.")
    
#diabetes prevalence and methane emissions
r, p_value = pearsonr(usdf['Diabetes prevalence'],usdf['Methane emissions'] )

# Print results
print(f"Pearson correlation coefficient: {r}")
print(f"p-value: {p_value}")

# Check if p-value is less than the significance level (e.g., 0.05)
if p_value < 0.05:
    print("The correlation is statistically significant.")
else:
    print("The correlation is not statistically significant.")
    
#adult male mortality rate and air pollution
r, p_value = pearsonr(usdf['Adult male mortality rate'],usdf['Air pollution'] )

# Print results
print(f"Pearson correlation coefficient: {r}")
print(f"p-value: {p_value}")

# Check if p-value is less than the significance level (e.g., 0.05)
if p_value < 0.05:
    print("The correlation is statistically significant.")
else:
    print("The correlation is not statistically significant.")

#life expectancy and air pollution
r, p_value = pearsonr(usdf['Life expectancy'],usdf['Air pollution'] )

# Print results
print(f"Pearson correlation coefficient: {r}")
print(f"p-value: {p_value}")

# Check if p-value is less than the significance level (e.g., 0.05)
if p_value < 0.05:
    print("The correlation is statistically significant.")
else:
    print("The correlation is not statistically significant.")
    
#methane emission and adult female mortality rate
r, p_value = pearsonr(usdf['Methane emissions'],usdf['Adult female mortality rate'] )

# Print results
print(f"Pearson correlation coefficient: {r}")
print(f"p-value: {p_value}")

# Check if p-value is less than the significance level (e.g., 0.05)
if p_value < 0.05:
    print("The correlation is statistically significant.")
else:
    print("The correlation is not statistically significant.")

#co2 damage and male survival
r, p_value = pearsonr(usdf['Carbon dioxide damage'],usdf['Male Survival to age 65'] )

# Print results
print(f"Pearson correlation coefficient: {r}")
print(f"p-value: {p_value}")

# Check if p-value is less than the significance level (e.g., 0.05)
if p_value < 0.05:
    print("The correlation is statistically significant.")
else:
    print("The correlation is not statistically significant.")
    
#freshwater withdrawals and infant mortality rate
r, p_value = pearsonr(usdf['Annual freshwater withdrawals'],usdf['Infant mortality rate'] )

# Print results
print(f"Pearson correlation coefficient: {r}")
print(f"p-value: {p_value}")

# Check if p-value is less than the significance level (e.g., 0.05)
if p_value < 0.05:
    print("The correlation is statistically significant.")
else:
    print("The correlation is not statistically significant.")
    
#methane emissions and adult male mortality rate
r, p_value = pearsonr(usdf['Methane emissions'],usdf['Adult male mortality rate'] )

# Print results
print(f"Pearson correlation coefficient: {r}")
print(f"p-value: {p_value}")

# Check if p-value is less than the significance level (e.g., 0.05)
if p_value < 0.05:
    print("The correlation is statistically significant.")
else:
    print("The correlation is not statistically significant.")
    
#net forest depletion and fertility rate
r, p_value = pearsonr(usdf['Net forest depletion'],usdf['Fertility rate'] )

# Print results
print(f"Pearson correlation coefficient: {r}")
print(f"p-value: {p_value}")

# Check if p-value is less than the significance level (e.g., 0.05)
if p_value < 0.05:
    print("The correlation is statistically significant.")
else:
    print("The correlation is not statistically significant.")
