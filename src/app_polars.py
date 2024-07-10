import streamlit as st
import polars as pl
from statsforecast import StatsForecast
import os

os.environ['NIXTLA_ID_AS_COL'] = '1'
from statsforecast.models import (
    HoltWinters, CrostonClassic, HistoricAverage,
    DynamicOptimizedTheta, SeasonalNaive, AutoARIMA,
    AutoETS, AutoTheta, AutoCES, ARIMA, MSTL
)
from statsforecast.feature_engineering import mstl_decomposition
import plotly.express as px

st.set_page_config(page_title="Auto Forecast Using Nixtla StatsForecast Library", layout="wide", page_icon="📈")

@st.cache_data
def load_data(_file):
    return pl.read_excel(_file, engine='xlsx2csv', read_options={"has_header": True})

@st.cache_data
def transform_data(_df):
    # Assume the first column is the ID column
    id_column = _df.columns[0]
    
    return (_df
            .unpivot(
                index=id_column,
                on=[col for col in _df.columns if col != id_column],
                variable_name='ds',
                value_name='value'
            )
            .with_columns(
                pl.col('ds').str.strptime(pl.Date, format="%Y-%m-%d %H:%M:%S")
            )
            .rename({
                id_column: "unique_id",
                "value": "y"
            })
            .select(['unique_id', 'ds', 'y'])
            .drop_nulls()
            .sort(['unique_id', 'ds'])
        )


@st.cache_data

def run_forecasts(_df, _transformed_df, _X_df, _models, _models2, freq, h, level, season_length):
    sf = StatsForecast(
        models=_models,
        freq=freq,
        n_jobs=-1,
        fallback_model=SeasonalNaive(season_length=season_length),
    )
    sf2 = StatsForecast(
        models=_models2,
        freq=freq,
        fallback_model=SeasonalNaive(season_length=season_length),
        n_jobs=-1,
    )
    
    forecasts = sf.forecast(df=_df, h=h, level=level)
    forecasts2 = sf2.forecast(df=_transformed_df, X_df=_X_df, h=h, level=level)
    
    # Merge the forecasts
    forecasts = forecasts.join(forecasts2, on=['unique_id', 'ds'], how='inner')
    return forecasts

def clear_cache():
    st.cache_data.clear()

def main():
    st.title("Auto Forecast Using Nixtla StatsForecast Library")

    with st.sidebar:    
        st.header("Settings")
        uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])
        freq = st.selectbox("Select data frequency", 
                            ["Daily", "Monthly", "Monthly", "Quarterly", "Yearly"])
        h = st.number_input("Forecast horizon", min_value=1, value=12)
        level = st.multiselect("Prediction interval levels", [80, 90, 95], default=[90])

    freq_map = {
        "Daily": "1d", "Monthly": "1mo", 
        "Quarterly": "1q", "Yearly": "1y"
    }
    selected_freq = freq_map[freq]
    season_length = {"1d": 7, "1mo": 12, "1q": 4, "1y": 1}[selected_freq]

    if uploaded_file:
        tab1, tab2, tab3 = st.tabs(["Forecasts", "Data Preview", "Transformed Data"])

        with tab2:
            df = load_data(uploaded_file)
            st.dataframe(df.head(10))  # Display only the first 10 rows
            st.text(f"Total rows: {len(df)}")

        with tab3:
            melted_df = transform_data(df)
            st.dataframe(melted_df.head(10))  # Display only the first 10 rows
            st.text(f"Total rows: {len(melted_df)}")
            st.download_button(
                label="Download Transformed Data as CSV",
                data=melted_df.write_csv(),
                file_name='transformed_data.csv',
                mime='text/csv',
            )
        with tab1:
            model = MSTL(season_length=season_length)
            transformed_df, X_df = mstl_decomposition(melted_df, model=model, freq=selected_freq, h=h)
        

            model_options = {
                "HoltWinters": HoltWinters(),
                "Croston": CrostonClassic(),
                "SeasonalNaive": SeasonalNaive(season_length=season_length),
                "HistoricAverage": HistoricAverage(),
                "DOT": DynamicOptimizedTheta(season_length=season_length),
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
        #clear cache
        clear_cache()
        with st.spinner('Generating forecasts...'):
            try:
                models = []
                models2 = []
                for model_name in selected_models:
                    if model_name == "AutoARIMA (MSTL)":
                        models2.append(model_options[model_name])
                    else:
                        models.append(model_options[model_name])
                forecasts_df = run_forecasts(melted_df, transformed_df, X_df, models, models2, selected_freq, h, level, season_length)
                
                # Store the forecasts in session state
                st.session_state.forecasts_df = forecasts_df
                
                st.success('Forecasts generated successfully!')
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

    # Check if forecasts exist in session state
    if 'forecasts_df' in st.session_state:
        forecasts_df = st.session_state.forecasts_df
        
        st.subheader("Forecast Summary")

        # Identify forecast columns (they will be named after the models)
        forecast_cols = [col for col in forecasts_df.columns if col not in ['unique_id', 'ds']]

        
        print(forecasts_df)
        # Allow user to view full results
        if st.checkbox("View Full Forecast Results"):
            st.dataframe(forecasts_df)

        # Plot the forecasts
        fig = StatsForecast.plot(melted_df, forecasts_df, level=level, engine='plotly')
        fig.update_layout(
            autosize=True,
            width=None,
            height=400,
            margin=dict(l=50, r=50, b=50, t=50, pad=4)
        )
        st.plotly_chart(fig, use_container_width=True)

        # Download button for forecast results
        st.download_button(
            label="Download Full Forecast Results as CSV",
            data=forecasts_df.write_csv(),
            file_name='forecast_results.csv',
            mime='text/csv',
        )   
             

    else:
        st.info("Please upload an Excel file to begin.")

    st.markdown("""
    <style>
    .stApp {
        background-color: #f0f2f6;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
    .stSelectbox>div>div>select {
        background-color: #f8f9fa;
    }
    </style>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
