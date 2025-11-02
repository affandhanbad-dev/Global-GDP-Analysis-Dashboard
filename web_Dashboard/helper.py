import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import pickle

class HelperClass:
    # helper functions....................................
    def basic_counts(df):
        region = df['Region'].nunique()
        countries = df['Country'].nunique()
        countries_counts = df['Region'].value_counts()
        return region, countries, countries_counts

    def ConvertToFloatAndFillMissValues(df):

        # convertion to float
        columns_to_keep_as_int = ['Population', 'Area (sq. mi.)']
        columns_to_skip = ['Region', 'Country'] + columns_to_keep_as_int

        for col in df.columns:
            if col not in columns_to_skip and df[col].dtype == 'O':
                df[col] = df[col].str.replace(',', '').astype(float)

        # fill miss values
        for col in df.columns.values:
            if df[col].isnull().sum() == 0:
                continue
            if col == 'Climate':
                guess_values = df.groupby('Region')['Climate'].apply(lambda x: x.mode().max())
            else:
                guess_values = df.groupby('Region')[col].median()
            for region in df['Region'].unique():
                df[col].loc[(df[col].isnull()) & (df['Region'] == region)] = guess_values[region]

        return df

    def AverageRegionsGDPLiteracyAgriculture(df):
        # Calculate the median for the specified columns by region
        result = df.groupby('Region')[['GDP ($ per capita)', 'Literacy (%)', 'Agriculture']].median()
        return result

        # Define a function to join all countries' data within each region

    def join_countries(data):
        return ', '.join(data.astype(str))

    def DataAgg(df):
        # Group the DataFrame by 'Region' and apply the join_countries function to aggregate country data
        region_data = df.groupby('Region').agg({
            'Country': HelperClass.join_countries,
            'Population': 'sum',
            'Area (sq. mi.)': 'sum',
            'Pop. Density (per sq. mi.)': 'mean',
            'Coastline (coast/area ratio)': 'mean',
            'Net migration': 'mean',
            'Infant mortality (per 1000 births)': 'mean',
            'GDP ($ per capita)': 'mean',
            'Literacy (%)': 'mean',
            'Phones (per 1000)': 'mean',
            'Arable (%)': 'mean',
            'Crops (%)': 'mean',
            'Other (%)': 'mean',
            'Climate': HelperClass.join_countries,
            'Birthrate': 'mean',
            'Deathrate': 'mean',
            'Agriculture': 'mean',
            'Industry': 'mean',
            'Service': 'mean'
        })

        # Reset the index to have 'Region' as a regular column
        region_data.reset_index(inplace=True)
        return region_data

    def top20_gdp_bar(df):
        try:
            plt.figure(figsize=(12, 8))
            plt.title("Top 20 Countries by GDP Per Capita")
            plt.xlabel("GDP ($ per capita)")
            plt.ylabel("Country")
            top20 = df.sort_values(by='GDP ($ per capita)', ascending=False).head(20)
            sns.barplot(
                x='GDP ($ per capita)',
                y='Country',
                data=top20,
                palette="Set1"
            )
            st.pyplot(plt)
        except KeyError as e:
            st.error(f"Column not found in dataset: {e}")
        except Exception as e:
            st.error(f"Error generating chart: {e}")

    def population_pie(df):
        try:
            region_data = df.groupby('Region')['Population'].sum()
            plt.figure(figsize=(7, 7))
            plt.pie(
                region_data,
                labels=region_data.index,
                autopct='%1.1f%%',
                startangle=140,
                colors=sns.color_palette("pastel")
            )
            plt.title('🌍 Population Distribution by Region', fontsize=14, fontweight='bold')
            plt.axis('equal')
            st.pyplot(plt)
        except KeyError as e:
            st.error(f"❌ Column not found in dataset: {e}")
        except Exception as e:
            st.error(f"⚠️ Error generating population chart: {e}")

    def correlation_heatmap(df):
        try:
            numeric_df = df.select_dtypes(include=np.number)
            correlation_matrix = numeric_df.corr()

            plt.figure(figsize=(12, 10))
            sns.heatmap(
                correlation_matrix,
                annot=True,
                cmap='coolwarm',
                fmt=".2f",
                linewidths=0.5,
                square=True
            )
            plt.title('🔗 Correlation Heatmap of Numerical Columns', fontsize=14, fontweight='bold')
            st.pyplot(plt)

        except Exception as e:
            st.error(f"⚠️ Error generating correlation heatmap: {e}")

    def AsiaFiveRegionGDP(df):
        top_five_asia_countries_literacy = df[df['Region'].str.strip() == 'ASIA (EX. NEAR EAST)'].nlargest(5,
                                                                                                           'Literacy (%)')
        top_five_asia_countries_literacy = top_five_asia_countries_literacy[
            ['Country', 'Literacy (%)', 'GDP ($ per capita)']]
        labels = top_five_asia_countries_literacy['Country']
        literacy_rates = top_five_asia_countries_literacy['Literacy (%)']
        gdp_values = top_five_asia_countries_literacy['GDP ($ per capita)']

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        # Create a pie chart for literacy rates
        axes[0].pie(literacy_rates, labels=labels, autopct='%1.1f%%', startangle=90)
        axes[0].set_title('Literacy Rates')

        # Create a pie chart for GDP per capita
        axes[1].pie(gdp_values, labels=labels, autopct='%1.1f%%', startangle=90)
        axes[1].set_title('GDP Per Capita')

        plt.tight_layout()
        plt.show()

        st.pyplot(fig)

    def literacy_pie(df):
        literacy_by_region = round(df.groupby('Region')['Literacy (%)'].mean(), 2)
        plt.figure(figsize=(7, 7))
        plt.pie(literacy_by_region, labels=literacy_by_region.index, autopct='%1.1f%%', startangle=140)
        plt.title('Literacy by Region')
        plt.axis('equal')
        st.pyplot(plt)

    def infant_mortality_pie(df):
        try:
            infant_mortality_by_region = round(
                df.groupby('Region')['Infant mortality (per 1000 births)'].mean(), 2
            )
            plt.figure(figsize=(7, 7))
            plt.pie(
                infant_mortality_by_region,
                labels=infant_mortality_by_region.index,
                autopct='%1.1f%%',
                startangle=140,
                colors=sns.color_palette('pastel')
            )
            plt.title('Infant Mortality Rate by Region')
            plt.axis('equal')
            st.pyplot(plt)
        except KeyError as e:
            st.error(f"Column not found in dataset: {e}")
        except Exception as e:
            st.error(f"Error generating pie chart: {e}")

    def EachReginGDP(df):
        # Group the DataFrame by 'Region' and calculate the mean GDP for each region
        region_gdp = df.groupby('Region')['GDP ($ per capita)'].mean()

        # Get the regions and mean GDP values
        regions = region_gdp.index
        mean_gdp = region_gdp.values

        # Calculate the number of subplots needed
        num_subplots = len(regions)
        num_cols = 5  # Set the number of columns for the grid

        # Calculate the number of rows needed
        num_rows = (num_subplots - 1) // num_cols + 1

        # Create the grid of subplots
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 4 * num_rows), constrained_layout=True)
        axes = axes.ravel()  # Flatten the 2D array of subplots

        # Customize parameters for better readability and spacing
        colors = plt.cm.tab20c(np.arange(20))
        autopct = '%1.1f%%'
        shadow = True

        for i in range(num_subplots):
            ax = axes[i]

            # Get the countries in the current region
            countries = df[df['Region'] == regions[i]]

            # Calculate the top 5 countries with the highest GDP in the current region
            top_countries = countries.nlargest(5, 'GDP ($ per capita)')

            # Get the top 5 countries and their mean GDP values
            country_names = top_countries['Country']
            country_gdp = top_countries['GDP ($ per capita)']

            # Generate colors for the top 5 countries
            region_colors = colors[:len(country_names)]

            # Define explode based on the number of countries in the region
            explode = [0] * len(country_names)

            ax.pie(country_gdp, labels=country_names, autopct=autopct, startangle=90,
                   colors=region_colors, shadow=shadow, explode=explode)
            ax.set_aspect('equal')  # Ensure the pie is drawn as a circle
            ax.set_title(f'{regions[i]} Region', fontsize=14)

        # Hide any remaining empty subplots
        for i in range(num_subplots, num_cols * num_rows):
            fig.delaxes(axes[i])

        # Add some space between the plots
        plt.subplots_adjust(wspace=0.5)

        # Show the pie charts
        plt.suptitle("Top 5 GDP Distribution by Region", fontsize=16)
        plt.show()

        st.pyplot(fig)