import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools as iter
import scipy.optimize as opt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score as silhouete_score

#2 files for the analysis
# file is about the population
file1 = 'total population.csv'
# this file is about the GDP
file2 = "GDP.csv"

def read(filename,format=1):
    """ Returning 2 data frames from the csv format file.The original dataframe is in the first image, and its transposed form is in the second.

        Parameters:

            filename: The csv filename.

        Returns:

            [DataFrame, Transposed DataFrame]: The initial dataframe

            and its rearranged counterpart."""

        
    if format:
        df = pd.read_csv(filename, skiprows=3)
    else:
        df = pd.read_excel(filename, skiprows=3)
    
    df.drop(df.columns[[1, 2, 3]], axis=1, inplace=True)
    return df, df.transpose()


# Reading the Total Population file in which population represents the original format and
# populationT represents the transposed version of the original data frame.

population, populationT = read(file1)
population

#Considering 1991 and 2021 for clustering analysis
year1 = '1991'
year2 = '2021'

interest_reg_list = ['Aruba', 'Afghanistan', 'Angola', 'Albania', 'Andorra',
       'United Arab Emirates', 'Argentina', 'Armenia', 'American Samoa',
       'Antigua and Barbuda', 'Australia', 'Austria', 'Azerbaijan',
       'Burundi', 'Belgium', 'Benin', 'Burkina Faso', 'Bangladesh',
       'Bulgaria', 'Bahrain', 'Bahamas, The', 'Bosnia and Herzegovina',
       'Belarus', 'Belize', 'Bermuda', 'Bolivia', 'Brazil', 'Barbados',
       'Brunei Darussalam', 'Bhutan', 'Botswana', 'Canada', 'Switzerland',
       'Channel Islands', 'Chile', 'China', "Cote d'Ivoire", 'Cameroon',
       'Congo, Rep.', 'Colombia', 'Comoros', 'Cabo Verde', 'Costa Rica',
       'Cuba', 'Curacao', 'Cayman Islands', 'Cyprus', 'Czechia',
       'Djibouti', 'Dominica', 'Denmark', 'Dominican Republic', 'Algeria',
       'Ecuador', 'Egypt, Arab Rep.', 'Eritrea', 'Spain', 'Estonia',
       'Ethiopia', 'Finland', 'Fiji', 'France', 'Faroe Islands',
       'Micronesia, Fed. Sts.', 'Gabon', 'United Kingdom', 'Georgia',
       'Ghana', 'Gibraltar', 'Guinea', 'Gambia, The', 'Guinea-Bissau',
       'Equatorial Guinea', 'Greece', 'Grenada', 'Greenland', 'Guatemala',
       'Guam', 'Guyana', 'Hong Kong SAR, China', 'Honduras', 'Croatia',
       'Haiti', 'Hungary', 'Indonesia', 'Isle of Man', 'India', 'Ireland',
       'Iran, Islamic Rep.', 'Iraq', 'Iceland', 'Israel', 'Italy',
       'Jamaica', 'Jordan', 'Japan', 'Kazakhstan', 'Kenya',
       'Kyrgyz Republic', 'Cambodia', 'Kiribati', 'St. Kitts and Nevis',
       'Korea, Rep.', 'Kuwait', 'Lao PDR', 'Lebanon', 'Liberia', 'Libya',
       'St. Lucia', 'Liechtenstein', 'Sri Lanka', 'Lesotho', 'Lithuania',
       'Luxembourg', 'Latvia', 'Macao SAR, China',
       'St. Martin (French part)', 'Morocco', 'Monaco', 'Moldova',
       'Madagascar', 'Maldives', 'Mexico', 'Marshall Islands',
       'North Macedonia', 'Mali', 'Malta', 'Myanmar', 'Montenegro',
       'Mongolia', 'Northern Mariana Islands', 'Mozambique', 'Mauritania',
       'Mauritius', 'Malawi', 'Malaysia', 'Namibia', 'New Caledonia',
       'Niger', 'Nigeria', 'Nicaragua', 'Netherlands', 'Norway', 'Nepal',
       'Nauru', 'New Zealand', 'Oman', 'Pakistan', 'Panama', 'Peru',
       'Philippines', 'Palau', 'Papua New Guinea', 'Poland',
       'Puerto Rico', "Korea, Dem. People's Rep.", 'Portugal', 'Paraguay',
       'Qatar', 'Romania', 'Russian Federation', 'Rwanda', 'Saudi Arabia',
       'Sudan', 'Senegal', 'Singapore', 'Solomon Islands', 'Sierra Leone',
       'El Salvador', 'San Marino', 'Somalia', 'Serbia', 'South Sudan',
       'Sao Tome and Principe', 'Suriname', 'Slovak Republic', 'Slovenia',
       'Sweden', 'Eswatini', 'Sint Maarten (Dutch part)', 'Seychelles',
       'Syrian Arab Republic', 'Turks and Caicos Islands', 'Chad', 'Togo',
       'Thailand', 'Tajikistan', 'Turkmenistan', 'Timor-Leste', 'Tonga',
       'Trinidad and Tobago', 'Tunisia', 'Turkiye', 'Tuvalu', 'Tanzania',
       'Uganda', 'Ukraine', 'Uruguay', 'United States', 'Uzbekistan',
       'St. Vincent and the Grenadines', 'Venezuela, RB',
       'British Virgin Islands', 'Virgin Islands (U.S.)', 'Vietnam',
       'Vanuatu', 'Samoa', 'Kosovo', 'Yemen, Rep.', 'South Africa',
       'Zambia', 'Zimbabwe']
# Taking the dataset of only  interested in regional
pop_cluster = population[population['Country Name'].isin(interest_reg_list)][['Country Name', year1, year2]].dropna()

# visualising data
plt.figure(dpi=140)
pop_cluster.plot(year1, year2, kind='scatter', color='red')
plt.title('Total Population')
plt.show()

# converting the dataframe of years to an array
p = pop_cluster[[year1, year2]].values

# taking  maximum and minimum value from arrays
max_value = p.max()
min_value = p.min()
# scaling the dataset using minimum and maximum value
scaled_data = (p - min_value) / (max_value - min_value)
print('\nNormalised Population:\n')

sse = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i,
                    max_iter=310, n_init=8, random_state=2,init='k-means++',)
    kmeans.fit(scaled_data)
    sse.append(kmeans.inertia_)

import warnings as warn
warn.filterwarnings(action='ignore')

# plotting to check for appropriate number of clusters using elbow method
plt.style.use('seaborn')
plt.figure(dpi=140)
plt.plot(range(1, 11,1), sse, color='blue')
plt.title('Elbow Method')
plt.xlabel('No. of clusters')
plt.ylabel('SSE')
plt.savefig('cluster.png')
plt.show()

# finding the Kmeans clusters
n_cluster = 2
kmeans = KMeans(n_clusters=n_cluster, max_iter=310,init='k-means++' , n_init=12, random_state=10)
# Fit the model to the data
kmeans.fit(scaled_data)

# Get labels
labels = kmeans.labels_

# Use the silhouette score to evaluate the quality of the clusters
print(f'Silhouette Score: {silhouete_score(scaled_data, labels)}')

# Extract cluster centers
centers = kmeans.cluster_centers_
print(centers)

# Plot scatter plot of clusters
plt.style.use('seaborn')
plt.figure(dpi=140)
plt.scatter(scaled_data[:, 0], scaled_data[:, 1], c=kmeans.labels_)
plt.title('K-Means Clustering')
plt.xlabel('1991')
plt.ylabel('2021')
plt.show()

# Get K-means
kmeans = KMeans(n_clusters=2, init='k-means++', n_init=12, random_state=20, max_iter=320)
y_predict = kmeans.fit_predict(scaled_data)
print(y_predict)

# Create new dataframe with labels for each country
pop_cluster['cluster'] = y_predict
pop_cluster.to_csv('clusters_results.csv', index=False)

# Plot normalized population
plt.style.use('seaborn')
plt.figure(dpi=140)
plt.scatter(scaled_data[y_predict == 0, 0], scaled_data[y_predict == 0, 1], s=50, c='blue', label='Cluster 0')
plt.scatter(scaled_data[y_predict == 1, 0], scaled_data[y_predict == 1, 1], s=50, c='green', label='Cluster 1')
plt.scatter(centers[:, 0], centers[:, 1], s=50, c='red', label='Centroids')
plt.title('Total Population (Normalized)')
plt.xlabel('1991')
plt.ylabel('2021')
plt.legend()
plt.show()
# Converting centroid to its unnormalized form
cent = centers * (max_value - min_value) + min_value

# Plotting the population in their clusters with the centroid points
plt.style.use('seaborn')
plt.figure(dpi=600)
plt.scatter(p[y_predict == 0, 0], p[y_predict == 0, 1],
            s=50, c='blue', label='Below 70 million')
plt.scatter(p[y_predict == 1, 0], p[y_predict == 1, 1],
            s=50, c='green', label='Above 70 million')
plt.scatter(cent[:, 0], cent[:, 1], s=50, c='red', label='Centroids')
plt.title('Total Population', fontsize=6)
plt.xlabel('1991', fontsize=6)
plt.ylabel('2021',  fontsize=6)
plt.legend()
plt.show()

# Curve Fitting Solutions Portion
# Examining the World Bank formatted Total Population file
population, population2 = read(file1)
population2.head(2)

# Renaming the transposed data columns for population
df2 = population2.rename(columns=population2.iloc[0])
# Dropping the country name
df2 = df2.drop(index=df2.index[0], axis=0)
df2['Year'] = df2.index

# Fitting the data for China's population
china = df2[['Year', 'China']].apply(pd.to_numeric, errors='coerce')
df_china = china.dropna()
print(df_china.head(20))

# Plotting the Total population in China year-wise
plt.figure(figsize=(16, 9))
df_china['China'].plot(
    kind='bar',
    color='red'
)
plt.title('Total population in China year-wise')
plt.show()

# Using the Logistic function for curve fitting and forecasting the Total Population
def logistic_func(current_t, initial_pop, growth_rate, inflection_point):
    """Estimating the logistic growth of a population .

    Current time is one of the parameters, current_t.
        initial_pop: The population at the start/initial.
        growth_rate: The rate of expansion/growth.
        the_inflection_point: The pivotal/inflection moment.

    returns: The population at the time in question and the growth rate g.
    """
    f = initial_pop / (1 + np.exp(-growth_rate * (current_t - inflection_point)))
    return f

# Doing Error ranges calculation
def err_ranges(inp_value, func, param, sigma):
    """ Determines the error bounds for a given function and its inputs.

        Parameters:
            inp_value: The function's input value.
            The function for which the error ranges will be determined and named is func.
            param: The function's parameters.
            Sigma: The data's standard deviation.

        The lower and upper error ranges are returned.
    """
    # Upper and Lower limits
    lower = func(inp_value, *param)
    upper = lower

    # Preparing upper and lower limits for parameters by creating the list of tuples
    up_low = []
    for p, s in zip(param, sigma):
        p_min = p - s
        p_max = p + s
        up_low.append((p_min, p_max))

    p_mix = list(iter.product(*up_low))

    # Calculate the upper and lower limits
    for p in p_mix:
        y = func(inp_value, *p)
        lower = np.minimum(lower, y)
        upper = np.maximum(upper, y)

    return lower, upper

# fits the logistic data
param_ch, covar_ch = opt.curve_fit(logistic_func, df_china['Year'], df_china['China'], p0=(3e12, 0.03, 2041))

# calculating the standard deviation
sigma_ch = np.sqrt(np.diag(covar_ch))
# Creating a new column with the fit data for China's population
df_china['fit'] = logistic_func(df_china['Year'], *param_ch)

# Forecast for the next 20 years for China's population
year = np.arange(1960, 2041)
forecast_ch = logistic_func(year, *param_ch)

# Calculating the error ranges for China's population
low_ch, up_ch = err_ranges(year, logistic_func, param_ch, sigma_ch)

# Plotting China's Total Population
plt.style.use('seaborn')
plt.figure(dpi=600)
plt.plot(df_china["Year"], df_china["China"], label="Population(China) ", c='purple')
plt.plot(year, forecast_ch, label="Forecast", c='red')
plt.fill_between(year, low_ch, up_ch, color="orange", alpha=0.7, label='Confidence Range ')
plt.xlabel("Year(China)", fontsize=6)
plt.ylabel("Population(China)", fontsize=6)
plt.legend()
plt.title('China', fontsize=6)
plt.show()

# Prints the error ranges for China's population
print(err_ranges(2041, logistic_func, param_ch, sigma_ch))


# Fitting the United States population data
USA = df2[['Year', 'United States']].apply(pd.to_numeric, errors='coerce')
USA = USA.dropna()

# Fits the US logistic data
param_usa, covar_usa = opt.curve_fit(logistic_func, USA['Year'], USA['United States'],
                                     p0=(3e12, 0.03, 2041))

# Calculating the standard deviation for United States data
sigma_usa = np.sqrt(np.diag(covar_usa))

# Forecast for the next 20 years for United States population
forecast_usa = logistic_func(year, *param_usa)

# Calculate error ranges for United States population
low_usa, up_usa = err_ranges(year, logistic_func, param_usa, sigma_usa)

# Plotting United State's Total Population
plt.style.use('seaborn')
plt.figure(dpi=600)
plt.plot(USA["Year"], USA["United States"], label="Population(United States)")
plt.plot(year, forecast_usa, label="Forecast", c='red')
plt.fill_between(year, low_usa, up_usa, color="orange", alpha=0.6, label="Confidence Range")
plt.xlabel("Year(United States)", fontsize=6)
plt.ylabel("Population(United States)", fontsize=6)
plt.legend(loc='upper left')
plt.title('USA Population', fontsize=6)
plt.show()

plt.figure(figsize=(16, 9))
USA['United States'].plot(kind='bar', color='red')
plt.title('Total population in United States Year Wise')
plt.show()
# fitting the data
ghana = df2[['Year', 'Ghana']].apply(pd.to_numeric, errors='coerce')
ghana = ghana.dropna()

# fits the Ghana's logistic data
param_ghana, covar_ghana = opt.curve_fit(logistic_func, ghana['Year'], ghana['Ghana'],
                                         p0=(3e12, 0.03, 2041))

# here sigma is the standard deviation
sigma_ghana = np.sqrt(np.diag(covar_ghana))

# Forecasting for the next 20 years
forecast_ghana = logistic_func(year, *param_ghana)

# Determining error ranges
low_ghana, up_ghana = err_ranges(year, logistic_func, param_ghana, sigma_ghana)

# createing a new column for the fit data in the Ghana dataframe
ghana['fit'] = logistic_func(ghana['Year'], *param_ghana)

# plotting Ghana's Total Population
plt.style.use('seaborn')
plt.figure(dpi=600)
plt.plot(ghana['Year'], ghana['Ghana'],
         label='Population(Ghana)', c='green')
plt.plot(year, forecast_ghana, label='Forecast', c='red')
plt.fill_between(year, low_ghana, up_ghana, color='orange',
                 alpha=0.7, label='Confidence Range')
plt.xlabel('Year(Ghana)',  fontsize=6)
plt.ylabel('Population(Ghana)',  fontsize=6)
plt.legend(loc='upper left')
plt.title('Ghana', fontsize=6)
plt.show()

# bar plot for Ghana's total population
plt.figure(figsize=(16, 9))
ghana['Ghana'].dropna().plot(kind='bar', color='red')
plt.title('Total Population in Ghana Year Wise')
plt.show()

#reading the World Bank format GDP/Capita file
gdp, gdpT = read(file2)

#rename the columns
gdp = gdpT.rename(columns=gdpT.iloc[0])

#drop the country name
gdp = gdp.drop(index=gdp.index[0], axis=0)
gdp['Year'] = gdp.index

#fitting the data
gdp_china = gdp[['Year', 'China']].apply(pd.to_numeric, 
                                               errors='coerce')
gdp_china=gdp_china.dropna()

# poly function for forecasting GDP/Capita
def poly(x, a, b, c):
    """ Determines the value of a polynomial function with the formula ax2 + bx + c.

        Parameters: x: The polynomial function's starting value.
            a: The polynomial's coefficient for x2.
            b: The polynomial's x coefficient.
            c: The polynomial's constant term.

        This function returns the polynomial function's value at x.

        """

    return a*x**2 + b*x + c

def get_error_estimates(x, y, degree):
    """ computes a polynomial function's error estimates.

       The data points' x values are the parameter x.
           y: The data points' y-values.
           degree: The polynomial's degree.

       Returns: The residuals' standard deviation as an estimate of the error. """

    coefficients = np.polyfit(x, y, degree)
    y_estimate = np.polyval(coefficients, x)
    residuals = y - y_estimate
# The residuals' standard deviation is used to estimate the error.

    return np.std(residuals)
# fits the linear data
param_china, cov_china = opt.curve_fit(poly, gdp_china['Year'], gdp_china['China'])

# calculates the standard deviation
sigma_china = np.sqrt(np.diag(cov_china))

# creates a new column for the fit figures
gdp_china['fit'] = poly(gdp_china['Year'], *param_china)

# forecasting the fit figures
forecast_china = poly(year, *param_china)

# error estimates
error_china = get_error_estimates(gdp_china['China'], gdp_china['fit'], 2)
print('\n Error Estimates for China GDP/Capita:\n', error_china)

# Plotting
plt.style.use('seaborn')
plt.figure(dpi=600)
plt.plot(gdp_china["Year"], gdp_china["China"], label="GDP/Capita", c='purple')
plt.plot(year, forecast_china, label="Forecast", c='red')

plt.xlabel("Year(China)",  fontsize=6)
plt.ylabel("Population(China)",  fontsize=6)
plt.legend()
plt.title('China', fontsize=6)
plt.show()
# Fitting the data
gdp_usa = gdp[['Year', 'United States']].apply(pd.to_numeric, errors='coerce')
gdp_usa = gdp_usa.dropna()

# Fits the linear data
param_usa, cov_usa = opt.curve_fit(poly, gdp_usa['Year'], gdp_usa['United States'])

# Calculates the standard deviation
sigma_usa = np.sqrt(np.diag(cov_usa))

# Creates a column for the fit data
gdp_usa['fit'] = poly(gdp_usa['Year'], *param_usa)

# Forecasting for the next 20 years
forecast_usa = poly(year, *param_usa)

# Error estimates
error_usa = get_error_estimates(gdp_usa['United States'], gdp_usa['fit'], 2)
print('\n Error Estimates for US GDP/Capita:\n', error_usa)

# Plotting
plt.style.use('seaborn')
plt.figure(dpi=600)
plt.plot(gdp_usa["Year"], gdp_usa["United States"], label="GDP/Capita")
plt.plot(year, forecast_usa, label="Forecast", c='red')
plt.xlabel("Year", fontsize=6)
plt.ylabel("GDP per Capita ('US$')",  fontsize=6)
plt.legend()
plt.title('United States',  fontsize=6)
plt.show()


#fitting the data
gdp_gh = gdp[['Year', 'Ghana']].apply(pd.to_numeric, 
                                               errors='coerce')
gdp_gh=gdp_gh.dropna()
#fits the linear data
param_gh, cov_gh = opt.curve_fit(poly, gdp_gh['Year'], gdp_gh['Ghana'])

#calculates the standard deviation
sigma_gh = np.sqrt(np.diag(cov_gh))

#creates a new column for the fit data
gdp_gh['fit'] = poly(gdp_gh['Year'], *param_gh)

#forescast paramaters for the next 20 years
forecast_gh = poly(year, *param_gh)

#error estimates
error_ghana = get_error_estimates(gdp_gh['Ghana'], gdp_gh['fit'], 2)
print('\n Error Estimates for Ghana GDP/Capita:\n', error_ghana)

#plotting
plt.style.use('seaborn')
plt.figure(dpi=600)
plt.plot(gdp_gh["Year"], gdp_gh["Ghana"], 
         label="GDP/Capita", c='green')
plt.plot(year, forecast_gh, label="Forecast", c='red')
plt.xlabel("Year", fontsize=6)
plt.ylabel("GDP per Capita ('Ghana')",  fontsize=6)
plt.legend()
plt.title('Ghana', fontsize=6)
plt.show()
