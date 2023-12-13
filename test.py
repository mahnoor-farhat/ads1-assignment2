# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 15:39:20 2023

@author: Mahnoor Farhat
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import kurtosis
from scipy.stats import skew


def population_data(filename):
    """
    This function reads the data from the csv file and results in two transposed dataframes.


    Parameters
    ----------
    filename : File path is given for the function to read the data from the file.

    """

    df1 = pd.read_csv(filename, skiprows=4)
    df1 = df1.dropna(axis=1, how='all')
    df1.set_index("Country Name", inplace=True)
    df1 = df1.dropna()
    df1 = df1.apply(pd.to_numeric, errors="coerce")
    
    #Dataframe transposed
    df_population_years = df1.T
    
    df_population_years = df_population_years.dropna()
    columns_to_drop = ['Country Code', 'Indicator Name', 'Indicator Code']
    df_population_countries = df1.drop(columns=columns_to_drop)
    return df_population_years, df_population_countries


filename = r"C:\Users\SAAD COMMUNICATION\OneDrive\Desktop\Maha UH Coursework\ADS 1 A2\population.csv"
df_population_years, df_population_countries = population_data(filename)

print("Dataframe with Years as Columns:\n\n", df_population_years.columns, "\n\n",
      "Dataframe with Countries as Columns:\n\n", df_population_countries.columns)


def energy_data(filename):
    """
    This function reads the data from the csv file and results in two transposed dataframes.


    Parameters
    ----------
    filename : File path is given for the function to read the data from the file.

    """
    
    df2 = pd.read_csv(filename, skiprows=4)
    df2 = df2.dropna(axis=1, how='all')
    df2.set_index("Country Name", inplace=True)
    df2 = df2.dropna()
    df2 = df2.apply(pd.to_numeric, errors="coerce")
    
    #Dataframe transposed
    df_energy_years = df2.T
    
    df_energy_years = df_population_years.dropna()
    columns_to_drop = ['Country Code', 'Indicator Name', 'Indicator Code']
    df_energy_countries = df2.drop(columns=columns_to_drop)
    return df_energy_years, df_energy_countries


filename = r"C:\Users\SAAD COMMUNICATION\OneDrive\Desktop\Maha UH Coursework\ADS 1 A2\energy_consumption.csv"
df_energy_years, df_energy_countries = population_data(filename)

print("Dataframe with Years as Columns:\n\n", df_energy_years.columns, "\n\n",
      "Dataframe with Countries as Columns:\n\n", df_energy_countries.columns)


def summary_statistics():
    """
    This function produces the summary statistics for the selected countries of a dataset.

    """
    
    selected_countries = ['United States', 'European Union', 'United Kingdom']
    df_selected1 = df_population_countries.loc[selected_countries]
    df_selected2 = df_energy_countries.loc[selected_countries]
    summary_stats1 = df_selected1.describe()
    summary_stats2 = df_selected2.describe()

    print("Summary Statistics for Annual Population Growth Data (US, EU, UK): \n")
    print(summary_stats1, "\n\n")

    print("Summary Statistics for Energy Consumption Data (US, EU, UK): \n")
    print(summary_stats2, "\n\n")



def subset_corr_matrix():
    """
    This function produces the a subset correlation matrix for two datasets.

    """

    df_population_transposed = df_population_years.T
    df_energy_transposed = df_energy_years.T
    correlation_matrix1 = df_population_transposed.corr()
    correlation_matrix2 = df_energy_transposed.corr()

    print("Correlation Matrix for Annual Population Growth: \n")
    print(correlation_matrix1, "\n\n")
    print("Correlation Matrix for Energy Consumption: \n")
    print(correlation_matrix2)

    subset_corr_matrix1 = correlation_matrix1.iloc[40:50, 40:50]
    subset_corr_matrix2 = correlation_matrix2.iloc[40:50, 40:50]

    plt.figure(figsize=(20, 10))
    sns.heatmap(subset_corr_matrix1, cmap="coolwarm",
                annot=True, fmt=".2f", linewidths=.5)
    plt.title("Subset Correlation Matrix for Population")
    plt.show()

    plt.figure(figsize=(20, 10))
    sns.heatmap(subset_corr_matrix2, cmap="coolwarm",
                annot=True, fmt=".2f", linewidths=.5)
    plt.title("Subset Correlation Matrix for Energy Consumption")
    plt.show()


def pop_kurt_skew():
    """
    This function produces the kurtosis and skewness values for the population growth dataset.

    """

    plt.hist([kurtosis(df_population_years), skew(df_population_years)], bins=10, color=['cyan', 'Purple'], edgecolor='black')

    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title('Distribution of Annual Population Growth Data')

    plt.legend(['kurtosis','skewness'])
    
    print("Skewness : \n", skew(df_population_years),"\n")
    print("Kurtosis : \n", kurtosis(df_population_years))


def energy_kurt_skew():
    """
    This function produces the kurtosis and skewness values for the energy consumption dataset.

    """
    
    plt.hist([kurtosis(df_energy_years), skew(df_energy_years)], bins=10, color=['orange', 'green'], edgecolor='black')

    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title('Distribution of Energy Consumption Data')

    plt.legend(['kurtosis','skewness'])

    print("Skewness : \n", skew(df_energy_years),"\n")
    print("Kurtosis : \n", kurtosis(df_energy_years))


def pop_line_graph():
    """
    This function reads the data from the csv file and plots a line graph.

    """

    df = pd.read_csv(r"C:\Users\SAAD COMMUNICATION\OneDrive\Desktop\Maha UH Coursework\ADS 1 A2\population.csv", skiprows=4)

    df_country1 = df[df['Country Name'] == 'Australia']
    df_years1 = df_country1.columns[4:]
    df_values1 = df_country1.iloc[:, 4:]

    df_country2 = df[df['Country Name'] == 'Japan']
    df_years2 = df_country2.columns[4:]
    df_values2 = df_country2.iloc[:, 4:]

    df_country3 = df[df['Country Name'] == 'European Union']
    df_years3 = df_country3.columns[4:]
    df_values3 = df_country3.iloc[:, 4:]

    df_country4 = df[df['Country Name'] == 'United Kingdom']
    df_years4 = df_country4.columns[4:]
    df_values4 = df_country4.iloc[:, 4:]

    df_country5 = df[df['Country Name'] == 'United States']
    df_years5 = df_country5.columns[4:]
    df_values5 = df_country5.iloc[:, 4:]

    df_country6 = df[df['Country Name'] == 'Middle East & North Africa']
    df_years6 = df_country6.columns[4:]
    df_values6 = df_country6.iloc[:, 4:]

    df_country7 = df[df['Country Name'] == 'China']
    df_years7 = df_country7.columns[4:]
    df_values7 = df_country7.iloc[:, 4:]


    plt.figure(figsize=(15, 6))

    plt.plot(df_years1, df_values1.values.ravel(), label = 'Australia')
    plt.plot(df_years2, df_values2.values.ravel(), label = 'Japan')
    plt.plot(df_years3, df_values3.values.ravel(), label = 'European Union')
    plt.plot(df_years4, df_values4.values.ravel(), label = 'United Kingdom')
    plt.plot(df_years5, df_values5.values.ravel(), label = 'United States')
    plt.plot(df_years6, df_values6.values.ravel(), label = 'Middle East & North Africa')
    plt.plot(df_years7, df_values7.values.ravel(), label = 'China')

    plt.xlabel('Year')
    plt.ylabel('Population Growth (%)')
    plt.title('Annual World Population Growth (Line Plot)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45) 

    plt.show()


def energy_line_graph():
    """
    This function reads the data from the csv file and plots a line graph.

    """
    
    df = pd.read_csv(r"C:\Users\SAAD COMMUNICATION\OneDrive\Desktop\Maha UH Coursework\ADS 1 A2\energy_consumption.csv", skiprows=4)

    df_country1 = df[df['Country Name'] == 'Australia']
    df_years1 = df_country1.columns[4:]
    df_values1 = df_country1.iloc[:, 4:]

    df_country2 = df[df['Country Name'] == 'Japan']
    df_years2 = df_country2.columns[4:]
    df_values2 = df_country2.iloc[:, 4:]

    df_country3 = df[df['Country Name'] == 'European Union']
    df_years3 = df_country3.columns[4:]
    df_values3 = df_country3.iloc[:, 4:]

    df_country4 = df[df['Country Name'] == 'United Kingdom']
    df_years4 = df_country4.columns[4:]
    df_values4 = df_country4.iloc[:, 4:]

    df_country5 = df[df['Country Name'] == 'United States']
    df_years5 = df_country5.columns[4:]
    df_values5 = df_country5.iloc[:, 4:]

    df_country6 = df[df['Country Name'] == 'Middle East & North Africa']
    df_years6 = df_country6.columns[4:]
    df_values6 = df_country6.iloc[:, 4:]

    df_country7 = df[df['Country Name'] == 'China']
    df_years7 = df_country7.columns[4:]
    df_values7 = df_country7.iloc[:, 4:]


    plt.figure(figsize=(15, 6))

    plt.plot(df_years1, df_values1.values.ravel(), label = 'Australia')
    plt.plot(df_years2, df_values2.values.ravel(), label = 'Japan')
    plt.plot(df_years3, df_values3.values.ravel(), label = 'European Union')
    plt.plot(df_years4, df_values4.values.ravel(), label = 'United Kingdom')
    plt.plot(df_years5, df_values5.values.ravel(), label = 'United States')
    plt.plot(df_years6, df_values6.values.ravel(), label = 'Middle East & North Africa')
    plt.plot(df_years7, df_values7.values.ravel(), label = 'China')

    plt.xlabel('Year')
    plt.ylabel('Oil Used per Capita (kg)')
    plt.title('World Energy Consumption (Line Plot)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45) 

    plt.show()


def pop_bar_graph():
    """
    This function reads the data from the csv file and plots a bar graph.

    """

    df = pd.read_csv(r"C:\Users\SAAD COMMUNICATION\OneDrive\Desktop\Maha UH Coursework\ADS 1 A2\population.csv", skiprows=4)

    df_year = df[df['1999'].notna()]
    df_sorted = df_year.sort_values(by='1999', ascending=False)
    df_top = 20
    df_top_countries = df_sorted.head(df_top)

    plt.figure(figsize=(12, 6))
    
    plt.bar(df_top_countries['Country Name'], df_top_countries['1999'], color = 'orange')

    plt.xlabel('Countries')
    plt.ylabel('Population Growth (%)')
    plt.title('Top 20 Countries with Highest Population Growth in 1999 (Bar Graph)')
    plt.xticks(rotation=45)  
    plt.tight_layout()

    plt.show()
    

def energy_bar_graph():
    """
    This function reads the data from the csv file and plots a bar graph.

    """
    
    df = pd.read_csv(r"C:\Users\SAAD COMMUNICATION\OneDrive\Desktop\Maha UH Coursework\ADS 1 A2\energy_consumption.csv", skiprows=4)

    df_year = df[df['1999'].notna()]
    df_sorted = df_year.sort_values(by='1999', ascending=False)
    df_top = 20
    df_top_countries = df_sorted.head(df_top)

    plt.figure(figsize=(12, 6))
    
    plt.bar(df_top_countries['Country Name'], df_top_countries['1999'], color = 'skyblue')

    plt.xlabel('Countries')
    plt.ylabel('Oil Used per Capita (kg)')
    plt.title('Top 20 Countries with Highest Energy Consumption in 1999 (Bar Graph)')
    plt.xticks(rotation=45)  
    plt.tight_layout()

    plt.show()
    
    
summary_statistics()
subset_corr_matrix()
pop_kurt_skew()
energy_kurt_skew()
pop_line_graph()
energy_line_graph()
pop_bar_graph()
energy_bar_graph()



