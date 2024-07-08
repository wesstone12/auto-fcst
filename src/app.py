import streamlit as st
import pandas as pd
from statsforecast import StatsForecast
import os
os.environ['NIXTLA_ID_AS_COL'] = '1'
from statsforecast.feature_engineering import mstl_decomposition
from statsforecast.models import ARIMA, MSTL
from statsforecast.models import (
    HoltWinters,
    CrostonClassic as Croston, 
    HistoricAverage,
    DynamicOptimizedTheta as DOT,
    SeasonalNaive, 
    AutoARIMA,
    AutoETS,
    AutoTheta,
    AutoCES
)
import plotly.express as px

st.set_page_config(page_title="Finance AutoML Data Preprocessor", layout="wide")

@st.cache_data
def load_data(file):
    df = pd.read_excel(file)
    df.columns = df.columns.map(str)
    return df

@st.cache_data
def transform_data(df):
    melted_df = pd.melt(df, id_vars=['Product'], var_name='ds', value_name='y')
    melted_df['unique_id'] = melted_df['Product']
    melted_df.drop(columns=['Product'], inplace=True)
    melted_df['ds'] = pd.to_datetime(melted_df['ds'], errors='coerce')
    melted_df = melted_df.dropna(subset=['ds'])
    melted_df = melted_df[['unique_id', 'ds', 'y']]
    melted_df = melted_df.sort_values(by=['unique_id', 'ds']).reset_index(drop=True)
    return melted_df

@st.cache_data
def run_forecasts(df, transformed_df, X_df, _models, _models2, freq, h, level, season_length):


    sf = StatsForecast(
        models=_models,
        freq=freq,
        fallback_model=SeasonalNaive(season_length=season_length),
        n_jobs=-1,
    )
    sf2 = StatsForecast(
        models=_models2,
        freq=freq,
        fallback_model=SeasonalNaive(season_length=season_length),
        n_jobs=-1,
    )
    
    forecasts_df = sf.forecast(df=df, h=h, level=level)
    forecasts_df2 = sf2.forecast(df=transformed_df, X_df=X_df, h=h, level=level)
    forecasts_df = pd.merge(forecasts_df, forecasts_df2, on=['unique_id', 'ds'])
    return forecasts_df

def clear_cache():
    st.cache_data.clear()

def main():
    st.title("Auto Forecast Using Nixtla StatsForecast Library  ")
    

    with st.sidebar:
        st.header("Settings")
        uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])
        freq = st.selectbox("Select the frequency of the data:", 
                            ["Daily", "Monthly (Start of the Month)", "Monthly (End of the Month)", "Quarterly", "Yearly"])
        h = st.number_input("Forecast horizon (number of periods):", min_value=1, value=12)
        level = st.multiselect("Select prediction interval levels:", [80, 90, 95], default=[90])

    freq_map = {
        "Daily": "D", "Monthly (Start of the Month)": "MS", "Monthly (End of the Month)": "ME",
        "Quarterly": "Q", "Yearly": "Y"
    }
    selected_freq = freq_map[freq]

    season_length_map = {"D": 7, "MS": 12, "ME": 12, "Q": 4, "Y": 1}
    season_length = season_length_map[selected_freq]

    if uploaded_file:
        tab1, tab2, tab3 = st.tabs(["Forecasts", "Data Preview", "Transformed Data", ])

        with tab2:
            st.header("Original Data")
            df = load_data(uploaded_file)
            st.write(df)

        with tab3:
            st.header("Transformed Data")
            melted_df = transform_data(df)
            st.write(melted_df)

            csv = melted_df.to_csv(index=False)
            st.download_button(
                label="Download Transformed Data as CSV",
                data=csv,
                file_name='transformed_data.csv',
                mime='text/csv',
            )


        with tab1:
            st.header("Forecasts")
            model = MSTL(season_length=season_length)
            transformed_df, X_df = mstl_decomposition(melted_df, model=model, freq=selected_freq, h=h)

            model_options = {
                "HoltWinters": HoltWinters(),
                "Croston": Croston(),
                "SeasonalNaive": SeasonalNaive(season_length=season_length),
                "HistoricAverage": HistoricAverage(),
                "DOT": DOT(season_length=season_length),
                "AutoARIMA": AutoARIMA(season_length=season_length, alias='AutoARIMA'),
                "AutoETS": AutoETS(season_length=season_length),
                "AutoTheta": AutoTheta(season_length=season_length),
                "AutoCES": AutoCES(season_length=season_length),
                "AutoARIMA (MSTL)": AutoARIMA(season_length=season_length, alias='AutoARIMA (MSTL)')
            }

            selected_models = st.multiselect("Select models to run:", 
                                 list(model_options.keys()), 
                                 default=list(model_options.keys()),
                                 key="model_selector")




            if st.button("Generate Forecasts"):
                st.success(f'Models selected: {selected_models}')
                #Clear cache
                clear_cache()

                with st.spinner('Generating forecasts...'):
                    try:
                        models = []
                        models2 = []
                        for model_name in selected_models:
                            if model_name == "AutoARIMA":
                                models2.append(model_options[model_name])
                            else:
                                models.append(model_options[model_name])

                        forecasts_df = run_forecasts(melted_df, transformed_df, X_df, models, models2, selected_freq, h, level, season_length)
                        st.success('Forecasts generated successfully!')
                        st.write(forecasts_df)
                       
                        

                        fig = StatsForecast.plot(melted_df, forecasts_df, level=level, engine='plotly')

                        # Adjust the layout for better responsiveness
                        fig.update_layout(
                            autosize=True,
                            width=None,
                            height=600,  # You can adjust this value as needed
                            margin=dict(l=50, r=50, b=100, t=100, pad=4)
                        )

                        # Display the chart using the full width of the container
                        st.plotly_chart(fig, use_container_width=True)

                        csv = forecasts_df.to_csv(index=False)
                        st.download_button(
                            label="Download Forecast Results as CSV",
                            data=csv,
                            file_name='forecast_results.csv',
                            mime='text/csv',
                        )
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")

        

    else:
        st.info("Please upload an Excel file to begin.")

    st.markdown("""
    <style>
    .stApp {
        background-color: #f0f2f6;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <style>
    div[data-testid="stMultiSelect"] > div > div > div {
        flex: 1 1 auto;
        width: auto;
        max-width: auto;
    }
    div[data-testid="stMultiSelect"] ul {
        width: auto;
        min-width: 400px;  /* Adjust this value as needed */
    }
    </style>
    """, unsafe_allow_html=True)
        

if __name__ == "__main__":
    main()