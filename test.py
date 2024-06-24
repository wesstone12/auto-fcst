import pandas as pd
import numpy as np
from tsfeatures import tsfeatures
import matplotlib.pyplot as plt
from fpdf import FPDF
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import io
from statsmodels.tsa.seasonal import seasonal_decompose
from PIL import Image
import multiprocessing
    # Load your data
from utilsforecast.data import generate_series
import matplotlib.dates as mdates
from PIL import Image

class TimeSeriesReport(FPDF):
    def __init__(self, df, features):
        super().__init__()
        self.df = df
        self.features = features
        self.set_auto_page_break(auto=True, margin=15)
        self.add_page()
        self.set_font("Arial", "B", 16)
        self.cell(0, 10, "Time Series Analysis Report", 0, 1, "C")
        self.ln(10)

    def time_series_plot(self):
        self.section_title("Time Series Plot")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.df['ds'], self.df['y'])
        ax.set_title("Time Series Plot")
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        plt.tight_layout()
        self.add_plot(fig)

    def section_title(self, title):
        self.set_font("Arial", "B", 14)
        self.cell(0, 10, title, 0, 1, "L")
        self.ln(5)

    def section_text(self, text):
        self.set_font("Arial", "", 12)
        self.multi_cell(0, 5, text)
        self.ln(5)

    

    def add_plot(self, fig):
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format='png', dpi=300)
        img_buffer.seek(0)
        
        # Convert to PIL Image
        pil_image = Image.open(img_buffer)
        
        # Save as temporary file
        temp_filename = f'temp_plot_{id(fig)}.png'
        pil_image.save(temp_filename)
        
        # Add image to PDF
        self.image(temp_filename, x=10, w=190)
        
        plt.close(fig)
        self.ln(5)
        
        # Remove temporary file
        import os
        os.remove(temp_filename)

    def basic_info(self):
        self.section_title("Basic Information")
        info_text = f"Series Length: {self.features['series_length'].values[0]}\n"
        
        if 'frequency' in self.features.columns:
            info_text += f"Frequency: {self.features['frequency'].values[0]}\n"
        else:
            info_text += "Frequency: Not available\n"
        
        info_text += f"Date Range: {self.df['ds'].min()} to {self.df['ds'].max()}"
        
        self.section_text(info_text)
    def descriptive_stats(self):
        self.section_title("Descriptive Statistics")
        stats = self.df['y'].describe()
        stats_text = (
            f"Mean: {stats['mean']:.2f}\n"
            f"Median: {stats['50%']:.2f}\n"
            f"Min: {stats['min']:.2f}\n"
            f"Max: {stats['max']:.2f}\n"
            f"Standard Deviation: {stats['std']:.2f}"
        )
        self.section_text(stats_text)

    def decomposition_plot(self):
        self.section_title("Time Series Decomposition")
        if 'frequency' in self.features.columns:
            period = max(int(self.features['frequency'].values[0]), 2)
        else:
            # If frequency is not available, try to infer it
            from statsmodels.tsa.seasonal import seasonal_decompose
            try:
                result = seasonal_decompose(self.df['y'], model='additive', extrapolate_trend='freq')
                period = result.seasonal.shape[0]
            except:
                self.section_text("Unable to perform time series decomposition due to insufficient seasonal information.")
                return

        try:
            stl = STL(self.df['y'], period=period)
            res = stl.fit()
            
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))
            
            ax1.plot(self.df['ds'], res.trend)
            ax1.set_title("Trend")
            ax1.set_xlabel("Date")
            ax1.set_ylabel("Value")
            
            ax2.plot(self.df['ds'], res.seasonal)
            ax2.set_title("Seasonal")
            ax2.set_xlabel("Date")
            ax2.set_ylabel("Value")
            
            ax3.plot(self.df['ds'], res.resid)
            ax3.set_title("Residual")
            ax3.set_xlabel("Date")
            ax3.set_ylabel("Value")
            
            for ax in (ax1, ax2, ax3):
                ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            plt.tight_layout()
            self.add_plot(fig)
        except Exception as e:
            self.section_text(f"Error in time series decomposition: {str(e)}")

    
    def acf_pacf_plot(self):
        self.section_title("Autocorrelation and Partial Autocorrelation")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        plot_acf(self.df['y'], lags=40, ax=ax1)
        ax1.set_title("Autocorrelation Function (ACF)")
        plot_pacf(self.df['y'], lags=40, ax=ax2)
        ax2.set_title("Partial Autocorrelation Function (PACF)")
        plt.tight_layout()
        self.add_plot(fig)

    def trend_analysis(self):
        self.section_title("Trend Analysis")
        if 'trend' in self.features.columns:
            trend_strength = self.features['trend'].values[0]
            trend_text = (
                f"Trend Strength: {trend_strength:.2f}\n\n"
                f"A trend strength of {trend_strength:.2f} suggests that "
                f"{'this time series has a strong overall direction' if trend_strength > 0.6 else 'the overall direction of this time series is not very strong'}. "
                "This means that the general pattern of the data "
                f"{'is moving consistently up or down over time' if trend_strength > 0.6 else 'does not show a consistent upward or downward movement over time'}."
            )
        else:
            trend_text = "Trend analysis not available."
        self.section_text(trend_text)

    def seasonality_analysis(self):
        self.section_title("Seasonality Analysis")
        if 'seasonal_strength' in self.features.columns:
            seasonal_strength = self.features['seasonal_strength'].values[0]
            seasonal_text = (
                f"Seasonal Strength: {seasonal_strength:.2f}\n\n"
                f"A seasonal strength of {seasonal_strength:.2f} indicates "
                f"{'strong' if seasonal_strength > 0.6 else 'moderate' if seasonal_strength > 0.3 else 'weak'} "
                "repeating patterns in your data. This could be daily, weekly, monthly, or yearly patterns "
                "depending on the frequency of your data."
            )
        else:
            seasonal_text = "Seasonal strength analysis not available."
        self.section_text(seasonal_text)

    def stationarity_analysis(self):
        self.section_title("Stationarity Analysis")
        kpss = self.features['unitroot_kpss'].values[0]
        pp = self.features['unitroot_pp'].values[0]
        stationarity_text = (
            f"KPSS Test Statistic: {kpss:.4f}\n"
            f"Phillips-Perron Test Statistic: {pp:.4f}\n\n"
            "Stationarity means that the statistical properties of a time series (like mean and variance) "
            "are constant over time. "
            f"Based on these tests, your time series appears to be "
            f"{'non-stationary' if kpss > 0.05 or pp > -3 else 'stationary'}. "
            f"{'This suggests that the series may need differencing or transformation before modeling.' if kpss > 0.05 or pp > -3 else 'This is generally good for forecasting.'}"
        )
        self.section_text(stationarity_text)

    def volatility_analysis(self):
        self.section_title("Volatility Analysis")
        arch_lm = self.features['arch_lm'].values[0]
        volatility_text = (
            f"ARCH LM Test Statistic: {arch_lm:.4f}\n\n"
            "Volatility refers to the degree of variation in the time series over time. "
            f"The ARCH LM test statistic of {arch_lm:.4f} suggests that "
            f"{'there is significant volatility clustering in the time series' if arch_lm > 0.05 else 'there is no significant volatility clustering in the time series'}. "
            f"{'This means that periods of high volatility tend to be followed by periods of high volatility, and vice versa.' if arch_lm > 0.05 else 'This means that the volatility is relatively constant over time.'}"
        )
        self.section_text(volatility_text)

    def generate_report(self):
        self.basic_info()
        self.descriptive_stats()
        self.time_series_plot()
        self.decomposition_plot()
        self.acf_pacf_plot()
        self.trend_analysis()
        self.seasonality_analysis()
        self.stationarity_analysis()
        self.volatility_analysis()
        self.output('time_series_report.pdf')

def create_time_series_report(df):
    # Ensure the DataFrame has the required columns
    assert all(col in df.columns for col in ['unique_id', 'ds', 'y']), "DataFrame must have 'unique_id', 'ds', and 'y' columns"

    # Extract features
    features = tsfeatures(df)
    
    print("Available features:", features.columns)

    # Create and generate the report
    report = TimeSeriesReport(df, features)
    report.generate_report()

    print("Report generated: time_series_report.pdf")

# Usage example:
# Assuming you have a DataFrame 'df' with columns 'unique_id', 'ds', and 'y'
# create_time_series_report(df)


def main():

    # Generate a time series
    series = generate_series(1, with_trend=True, static_as_categorical=False)
    print(series)
    
    # Call your report creation function
    create_time_series_report(series)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()