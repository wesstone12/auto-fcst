import streamlit as st
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import (
    HoltWinters,
    CrostonClassic as Croston, 
    HistoricAverage,
    DynamicOptimizedTheta as DOT,
    SeasonalNaive
)
import matplotlib.pyplot as plt

def main():
    st.title("Finance AutoML Data Preprocessor")

    uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

    if uploaded_file:
        # Read the Excel file and convert column names to strings
        df = pd.read_excel(uploaded_file)
        df.columns = df.columns.map(str)
        st.write("Original Data:")
        st.write(df)

        # Melt the data
        melted_df = pd.melt(df, id_vars=['Product'], var_name='ds', value_name='y')

        # Generate unique_id
        melted_df['unique_id'] = melted_df['Product']
        melted_df.drop(columns=['Product'], inplace=True)

        # Convert 'ds' to datetime format
        melted_df['ds'] = pd.to_datetime(melted_df['ds'], errors='coerce')

        # Check for any NaT values in 'ds' and drop them
        melted_df = melted_df.dropna(subset=['ds'])

        # Reorder columns
        melted_df = melted_df[['unique_id', 'ds', 'y']]

        # Sort values
        melted_df = melted_df.sort_values(by=['unique_id', 'ds']).reset_index(drop=True)

        st.write("Transformed Data:")
        st.write(melted_df)

        # Select frequency
        freq = st.selectbox("Select the frequency of the data:", ["Daily", "Monthly", "Quarterly", "Yearly"])
        freq_map = {
            "Daily": "D",
            "Monthly": "M",
            "Quarterly": "Q",
            "Yearly": "Y"
        }
        selected_freq = freq_map[freq]

        # Determine season length based on frequency
        season_length_map = {
            "D": 7,  # Weekly seasonality
            "M": 12, # Yearly seasonality
            "Q": 4,  # Yearly seasonality
            "Y": 1   # Yearly seasonality
        }
        season_length = season_length_map[selected_freq]

        # Define models with dynamic season length
        models = [
            HoltWinters(),
            Croston(),
            SeasonalNaive(season_length=season_length),
            HistoricAverage(),
            DOT(season_length=season_length)
        ]

        # Instantiate StatsForecast
        sf = StatsForecast(
            models=models,
            freq=selected_freq,
            fallback_model=SeasonalNaive(season_length=season_length),
            n_jobs=-1,
        )

        # Fit the models
        sf.fit(melted_df)

        # Forecast horizon (can be modified based on requirements)
        h = st.number_input("Forecast horizon (number of periods):", min_value=1, value=12)
        level = st.multiselect("Select prediction interval levels:", [80, 90, 95], default=[90])

        if st.button("Forecast"):
            forecasts_df = sf.forecast(df=melted_df, h=h, level=level)
            st.write("Forecasted Data:")
            st.write(forecasts_df)

            # Plot the results
            fig = sf.plot(melted_df, forecasts_df)
            st.pyplot(fig)

        # Option to download the transformed data
        csv = melted_df.to_csv(index=False)
        st.download_button(
            label="Download Transformed Data as CSV",
            data=csv,
            file_name='transformed_data.csv',
            mime='text/csv',
        )

if __name__ == "__main__":
    main()
