import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.optimize import curve_fit
from statsmodels.tsa.stattools import adfuller, kpss
from arch.unitroot import PhillipsPerron
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU
from sklearn.metrics import mean_squared_error
import warnings
import os
import seaborn as sns
import re

# --- Notebook Functions ---

def acf_plot_st(series, lags=30, title='Autocorrelation Function (ACF)'):
    """Generates an ACF plot for a Streamlit app."""
    fig, ax = plt.subplots(figsize=(15,4))
    plot_acf(series, lags=lags, ax=ax)
    ax.set_title(title)
    return fig

def pacf_plot_st(series, lags=30, title='Partial Autocorrelation Function (PACF)'):
    """Generates a PACF plot for a Streamlit app."""
    fig, ax = plt.subplots(figsize=(15,4))
    plot_pacf(series, lags=lags, ax=ax)
    ax.set_title(title)
    return fig

def health_indicator_fn(bearing_data, use_filter=False):
    """Calculates the health indicator (PC1) using PCA with error handling."""
    data = bearing_data.copy()
    if use_filter:
        for ft in data.columns:
            if data[ft].isnull().any():
                data[ft] = data[ft].fillna(method='bfill').fillna(method='ffill')
            if data[ft].isnull().any():
                 data[ft] = data[ft].fillna(0)
            data[ft] = data[ft].ewm(span=40,adjust=False).mean()

    data = data.fillna(method='bfill').fillna(method='ffill')
    if data.isnull().any().any():
        data = data.fillna(0)

    pca = PCA()
    if data.empty or data.isnull().values.any():
        st.warning("Data for PCA is empty or contains NaNs. Returning empty DataFrame.")
        return pd.DataFrame(columns=['PC1', 'cycle']), 0.0

    X_pca = pca.fit_transform(data)
    component_names = [f"PC{i+1}" for i in range(X_pca.shape[1])]
    X_pca = pd.DataFrame(X_pca, columns=component_names)
    
    explained_variance_ratio_pc1 = 0.0
    if pca.explained_variance_ratio_ is not None and len(pca.explained_variance_ratio_) > 0:
        explained_variance_ratio_pc1 = pca.explained_variance_ratio_[0]
    
    if 'PC1' not in X_pca.columns:
        st.warning("PC1 not found after PCA. Returning empty DataFrame.")
        return pd.DataFrame(columns=['PC1', 'cycle']), explained_variance_ratio_pc1

    health_indicator_pc1 = np.array(X_pca['PC1'])
    degredation = pd.DataFrame(health_indicator_pc1, columns=['PC1'])
    degredation['cycle'] = degredation.index
    
    if not degredation['PC1'].empty:
        # Ensure the HI trend is increasing (representing degradation)
        if degredation['PC1'].iloc[-1] < degredation['PC1'].iloc[0]:
            degredation['PC1'] = -degredation['PC1']
        degredation['PC1'] = degredation['PC1'] - degredation['PC1'].min(axis=0)
    else:
        st.warning("PC1 data is empty. Returning as is.")

    return degredation, explained_variance_ratio_pc1


def exp_fit_fn(df, base=500, print_parameters=False):
    """Performs an exponential fit on the degradation data."""
    x =np.array(df.cycle)
    if len(x) < base:
        base = len(x)
    if base < 2: # curve_fit needs at least 2 points
        st.warning("Not enough data points for exponential fit.")
        return (np.array([0,0]), None)

    x_fit = x[-base:].copy()
    y_fit = np.array(df.PC1)
    y_fit = y_fit[-base:].copy()

    def exp_func(x_val,a,b):
        y_val = a*np.exp(abs(b)*x_val)
        return y_val
    
    try:
        fit_params, cov = curve_fit(exp_func, x_fit, y_fit, p0=[0.01,0.001], maxfev=10000)
        if print_parameters:
            st.write("Fit Parameters (a,b):", fit_params)
        return fit_params, cov
    except RuntimeError:
        st.warning("Exponential fit could not converge.")
        return (np.array([0.01, 0.001]), None)


def predict_fn(X_df, p_tuple):
    """Predicts using the exponential fit parameters."""
    if p_tuple[0] is None:
         st.warning("Fit parameters are None. Cannot predict.")
         return np.zeros(len(X_df.cycle))

    x =np.array(X_df.cycle)
    a,b = p_tuple[0]
    fit_eq = a*np.exp(abs(b)*x)
    return fit_eq

def fenetre_fn(df_vals, w):
    """Creates a windowed dataset for time series forecasting."""
    if isinstance(df_vals, pd.Series):
      df_vals = df_vals.values
    if df_vals.ndim > 1:
      df_vals = df_vals.flatten()

    if len(df_vals) < w:
        return np.array([]).reshape(0, w), np.array([])

    windowed_data = np.array([df_vals[i:i+w] for i in range(len(df_vals) - w + 1)])
    if windowed_data.shape[0] == 0:
        return np.array([]).reshape(0, w-1), np.array([])
    return windowed_data[:, :-1], windowed_data[:, -1]


def test_ADF_st(serie):
  """Performs and displays ADF test in Streamlit."""
  resultat=adfuller(serie)
  st.subheader("Augmented Dickey-Fuller (ADF) Test")
  st.write(f"**ADF Statistic:** `{resultat[0]:.4f}`")
  st.write(f"**P-value:** `{resultat[1]:.4f}`")
  st.write("**Critical Values:**")
  st.json({k: f"{v:.4f}" for k, v in resultat[4].items()})
  if resultat[1] > 0.05:
    st.error("Result: The p-value is greater than 0.05. The series is likely non-stationary.")
  else:
    st.success("Result: The p-value is less than 0.05. The series is likely stationary.")

def test_PP_st(serie):
  """Performs and displays Phillips-Perron test in Streamlit."""
  PP_test=PhillipsPerron(serie)
  st.subheader("Phillips-Perron (PP) Test")
  st.write(f"**Test Statistic:** `{PP_test.stat:.4f}`")
  st.write(f"**P-value:** `{PP_test.pvalue:.4f}`")
  st.write(f"**Lags:** `{PP_test.lags}`")
  if PP_test.pvalue > 0.05:
    st.error("Result: The p-value is greater than 0.05. The series is likely non-stationary.")
  else:
    st.success("Result: The p-value is less than 0.05. The series is likely stationary.")

def test_KPSS_st(serie):
    """Performs and displays KPSS test in Streamlit."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        kpss_test = kpss(serie, regression="c", nlags="auto")
    st.subheader("Kwiatkowski-Phillips-Schmidt-Shin (KPSS) Test")
    st.write(f"**Test Statistic:** `{kpss_test[0]:.4f}`")
    st.write(f"**p-value:** `{kpss_test[1]:.4f}`")

    if kpss_test[1] < 0.05:
        st.error("Result: The p-value is less than 0.05. The series is likely non-stationary.")
    else:
        st.success("Result: The p-value is greater than 0.05. The series is likely stationary.")

# --- Streamlit App ---

st.set_page_config(layout="wide")
st.title("Bearing Time Series Analysis and RUL Prediction")

# --- Data Loading and Caching ---
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        try:
            df_loaded = pd.read_csv(uploaded_file)
            df_loaded = df_loaded.rename(columns={'Unnamed: 0':'time'})
            return df_loaded
        except Exception as e:
            st.error(f"Error loading CSV: {e}")
            return None
    return None

uploaded_file = st.sidebar.file_uploader("Upload 'features_1st_test.csv'", type="csv")

if uploaded_file is None:
    st.info("Please upload the 'features_1st_test.csv' file to begin.")
    st.stop()

set1 = load_data(uploaded_file)

if set1 is None:
    st.stop()

last_cycle_val = len(set1)

# --- Sidebar Controls ---
st.sidebar.header("Configuration")
selected_bearing = st.sidebar.selectbox("Select Bearing (B)", [1, 2, 3, 4], index=2)
init_cycle_pca = st.sidebar.slider("Initial Cycles for PCA data slice", 100, last_cycle_val - 50 , 600)
use_filter_pca = st.sidebar.checkbox("Use EMA Filter for PCA Health Indicator", True)

prediction_cycle_interactive = st.sidebar.slider("Current Cycle for RUL Plot", init_cycle_pca, last_cycle_val, init_cycle_pca, 25)
base_for_exp_fit = st.sidebar.slider("Data points for Exponential Fit (base)", 50, 500, 250, 50)
failure_threshold_rul = st.sidebar.number_input("Failure Threshold for RUL", value=2.0, step=0.1)


# --- Main App Sections using Tabs ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Data Overview & Correlation",
    "Health Indicator & Exponential RUL",
    "Statistical Tests",
    "ARIMA Modeling",
    "Neural Networks",
    "Chatbot"
])


with tab1:
    st.header("Uploaded Data Overview")
    st.dataframe(set1.head())
    st.write(f"Dataset shape: {set1.shape}")

    st.header("Feature Correlation Matrix")
    st.info("This heatmap shows the correlation between all the numeric features in the dataset. Values close to 1 (red) or -1 (blue) indicate a strong positive or negative linear relationship, respectively.")
    
    numeric_cols = set1.select_dtypes(include=np.number).columns
    corr = set1[numeric_cols].corr()
    
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(corr, annot=False, cmap='coolwarm', ax=ax)
    ax.set_title("Correlation Matrix of Bearing Features")
    st.pyplot(fig)


# Define features for HI (used in multiple tabs)
selected_features_pca = ['max','p2p','rms']
B_x_features = [f"B{selected_bearing}_x_{i}" for i in selected_features_pca]
# Validate features once
if not all(feature in set1.columns for feature in B_x_features):
    st.error(f"One or more default features for Bearing B{selected_bearing} not found.")
    st.stop()
    
degredation_0_full, _ = health_indicator_fn(set1[B_x_features], use_filter=use_filter_pca)


with tab2:
    st.header(f"Health Indicator (PCA) and RUL Prediction for Bearing B{selected_bearing}")
    
    early_cycles_data_pca = set1[B_x_features][:init_cycle_pca]
    early_cycles_pca_df, explained_var_pca = health_indicator_fn(early_cycles_data_pca, use_filter=use_filter_pca)

    st.write(f"**PCA Health Indicator (using data up to cycle {init_cycle_pca})**")
    st.write(f"Percentage of variance explained by PC1: {explained_var_pca:.4f}")
    
    fig_pca_hi, ax_pca_hi = plt.subplots(figsize=(10,5))
    ax_pca_hi.plot(early_cycles_pca_df['cycle'], early_cycles_pca_df['PC1'])
    ax_pca_hi.set_title(f'Health Indicator (PC1) for Bearing B{selected_bearing}')
    ax_pca_hi.set_xlabel('Cycle'); ax_pca_hi.set_ylabel('PC1 (Health Indicator)'); st.pyplot(fig_pca_hi); plt.close(fig_pca_hi)

    st.write(f"**RUL Prediction (Exponential Fit) at Cycle {prediction_cycle_interactive}**")
    data_for_rul_plot = set1[B_x_features][:prediction_cycle_interactive]
    
    if data_for_rul_plot.shape[0] > 2:
        degredation_for_rul_plot, _ = health_indicator_fn(data_for_rul_plot, use_filter=use_filter_pca)
        if not degredation_for_rul_plot.empty and len(degredation_for_rul_plot) >= base_for_exp_fit:
            fit_params_rul, _ = exp_fit_fn(degredation_for_rul_plot, base=base_for_exp_fit)
            if fit_params_rul is not None and len(fit_params_rul) == 2:
                prediction_hi_values = predict_fn(degredation_for_rul_plot, (fit_params_rul, _))
                m_rul, n_rul = fit_params_rul
                fail_cycle_rul = (np.log(failure_threshold_rul / m_rul) / abs(n_rul)) if m_rul > 0 and abs(n_rul) > 1e-9 else float('inf')
                st.write(f"Predicted Fail Cycle: {fail_cycle_rul:.2f}")
                fig_rul, ax_rul = plt.subplots(figsize=(10,5)); ax_rul.plot([0,prediction_cycle_interactive],[failure_threshold_rul, failure_threshold_rul],label='Threshold', color='k', ls='--')
                ax_rul.scatter(degredation_for_rul_plot['cycle'],degredation_for_rul_plot['PC1'],color='b',s=10,label='Actual HI')
                ax_rul.plot(degredation_for_rul_plot['cycle'],prediction_hi_values,color='r',alpha=0.7,label='Predicted HI'); ax_rul.legend(); st.pyplot(fig_rul); plt.close(fig_rul)

with tab3:
    st.header(f"Statistical Stationarity Tests for HI of Bearing B{selected_bearing}")
    if not degredation_0_full.empty and not degredation_0_full['PC1'].isnull().all():
        pc1_series = degredation_0_full['PC1']
        st.markdown("### Tests on Original HI (PC1) Series"); col1, col2, col3 = st.columns(3)
        with col1: test_ADF_st(pc1_series)
        with col2: test_PP_st(pc1_series)
        with col3: test_KPSS_st(pc1_series)
        st.markdown("---")
        st.markdown("### Tests on Differenced HI (PC1) Series")
        degredation_1_diff = pc1_series.diff().dropna()
        if not degredation_1_diff.empty:
            col1_d, col2_d, col3_d = st.columns(3)
            with col1_d: test_ADF_st(degredation_1_diff)
            with col2_d: test_PP_st(degredation_1_diff)
            with col3_d: test_KPSS_st(degredation_1_diff)

with tab4:
    st.header(f"ARIMA Modeling for HI of Bearing B{selected_bearing}")
    if not degredation_0_full.empty and not degredation_0_full['PC1'].isnull().all():
        pc1_series_full = degredation_0_full['PC1']
        degredation_1_diff_full = pc1_series_full.diff().dropna()
        if len(degredation_1_diff_full) > 50:
            st.pyplot(acf_plot_st(degredation_1_diff_full)); st.pyplot(pacf_plot_st(degredation_1_diff_full))
            st.subheader("ARIMA Model Parameters"); col_p, col_d, col_q = st.columns(3)
            p = col_p.number_input('AR (p)', 0, 10, 2); d = col_d.number_input('I (d)', 0, 5, 1); q = col_q.number_input('MA (q)', 0, 10, 2)
            if st.button("Run ARIMA Model"):
                series_to_fit = pc1_series_full if d > 0 else degredation_1_diff_full
                effective_d = d if d > 0 else 0 
                split_pt = int(len(series_to_fit) * 0.9); train, test = series_to_fit[:split_pt], series_to_fit[split_pt:]
                if len(train) > 10 and len(test) > 1:
                    try:
                        with st.spinner(f"Fitting ARIMA({p}, {effective_d}, {q})..."):
                            model_arima=ARIMA(train, order=(p,effective_d,q)).fit()
                        with st.expander("ARIMA Model Summary"): st.text(model_arima.summary().as_text())
                        preds=model_arima.predict(start=len(train), end=len(series_to_fit)-1)
                        st.write(f"MSE: {mean_squared_error(test, preds):.6f}")
                        fig, ax = plt.subplots(figsize=(12, 6)); ax.plot(test.index, test.values, label='Actual'); ax.plot(preds.index, preds.values, color='red', label='Predicted'); ax.legend(); st.pyplot(fig)
                    except Exception as e: st.error(f"ARIMA Error: {e}")

with tab5:
    st.header("Neural Network RUL Prediction")
    if not degredation_0_full.empty and not degredation_0_full['PC1'].isnull().all():
        pc1_vals_nn = degredation_0_full['PC1'].values; w_nn = 4
        if len(pc1_vals_nn) < w_nn + 10: st.warning("Not enough data for NN modeling.")
        else:
            X_nn, y_nn = fenetre_fn(pc1_vals_nn, w_nn)
            if X_nn.shape[0] > 0:
                y_nn = y_nn.reshape((len(y_nn),1)); split_pt = int(len(X_nn) * 0.9)
                X_train, X_test = X_nn[:split_pt,:], X_nn[split_pt:,:]
                Y_train, Y_test = y_nn[:split_pt], y_nn[split_pt:]
                X_train_rec = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
                X_test_rec = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
                def get_and_predict_rul(model_name, get_model_fn, model_filename):
                    st.subheader(model_name)
                    if st.button(f"Calculate RUL with {model_name}"):
                        try:
                            model = get_model_fn((X_train_rec.shape[1], X_train_rec.shape[2]))
                            model_path = f"{model_filename}_{selected_bearing}.h5"
                            if not os.path.exists(model_path):
                                with st.spinner(f"Training new {model_name} model..."):
                                    model.compile(optimizer='adam', loss='mse')
                                    model.fit(X_train_rec, Y_train, epochs=50, batch_size=16, verbose=0)
                                    model.save_weights(model_path)
                                st.success(f"Model trained and saved to `{model_path}`.")
                            else: 
                                model.load_weights(model_path); st.success(f"Loaded pre-trained model.")
                            with st.spinner("Forecasting..."):
                                last_seq = X_test_rec[-1:].copy(); future_preds = []; rul_cycles = -1
                                for i in range(5000):
                                    next_pred = model.predict(last_seq, verbose=0)[0,0]
                                    if next_pred >= failure_threshold_rul: rul_cycles = i+1; break
                                    future_preds.append(next_pred)
                                    last_seq = np.append(last_seq[0,:,0][1:], next_pred).reshape(1, w_nn-1, 1)
                            st.subheader("RUL Prediction Results")
                            if rul_cycles != -1:
                                r_h = (rul_cycles*10)/60; r_d = r_h/24
                                st.session_state['last_rul_prediction'] = {"model":model_name,"b":selected_bearing,"c":rul_cycles,"h":r_h,"d":r_d}
                                c1,c2,c3 = st.columns(3)
                                c1.metric("RUL (Cycles)", f"{rul_cycles}"); c2.metric("RUL (Hours)", f"{r_h:.2f}"); c3.metric("RUL (Days)", f"{r_d:.2f}")
                                fig,ax = plt.subplots(figsize=(10,5)); hist_cyc = np.arange(len(pc1_vals_nn)-50, len(pc1_vals_nn))
                                ax.plot(hist_cyc, pc1_vals_nn[-50:], label="Recent HI"); fut_cyc = np.arange(len(pc1_vals_nn), len(pc1_vals_nn)+len(future_preds)+1)
                                ax.plot(fut_cyc, np.append(pc1_vals_nn[-1], future_preds), label="Forecasted HI", ls='--')
                                ax.axhline(y=failure_threshold_rul, color='r', ls='-', label='Threshold'); ax.legend(); st.pyplot(fig)
                            else: st.warning("Failure not predicted within forecast horizon.")
                        except Exception as e: st.error(f"An error occurred: {e}")
                
                c1,c2,c3=st.columns(3)
                with c1: get_and_predict_rul("Simple RNN", lambda s:Sequential([SimpleRNN(32,input_shape=s),Dense(1)]), "rnn.h5")
                with c2: get_and_predict_rul("LSTM", lambda s:Sequential([LSTM(32,input_shape=s),Dense(1)]), "lstm.h5")
                with c3: get_and_predict_rul("GRU", lambda s:Sequential([GRU(32,input_shape=s),Dense(1)]), "gru.h5")

# --- Chatbot Logic ---
def get_chatbot_response(prompt):
    """A more advanced rule-based chatbot using a knowledge base."""
    prompt = prompt.lower()
    tokens = set(re.findall(r'\b\w+\b', prompt))

    knowledge_base = {
        ('rul', 'prediction', 'result', 'latest'): "get_last_prediction",
        ('how', 'work', 'app'): "This app analyzes bearing vibration data. You can upload your data, and it generates a Health Indicator (HI) using PCA. You can then use various models like ARIMA and Neural Networks in their respective tabs to predict the Remaining Useful Life (RUL).",
        ('guide', 'help', 'do'): "I can provide information about the last RUL prediction or explain concepts like PCA and ARIMA. Try asking 'What was the last result?' or 'What is PCA?'. To predict RUL, go to the 'Neural Networks' or 'ARIMA Modeling' tab.",
        ('what', 'pca'): "PCA (Principal Component Analysis) is a technique used to reduce the dimensionality of data. Here, we use it to combine multiple sensor features into a single Health Indicator (HI) that captures the main trend of degradation.",
        ('what', 'arima'): "ARIMA is a statistical model used for time series forecasting. It stands for AutoRegressive Integrated Moving Average. You can experiment with its parameters (p,d,q) in the 'ARIMA Modeling' tab.",
        ('what', 'stationarity', 'stationary'): "A time series is stationary if its statistical properties (like mean and variance) do not change over time. Non-stationary series are unpredictable. You can check for stationarity in the 'Stat. Tests' tab. Models like ARIMA often require the data to be stationary.",
        ('health', 'indicator', 'hi'): "The Health Indicator (HI) is a single value derived from multiple sensor features using PCA. It's designed to represent the overall health of the bearing over time. An increasing HI value signifies degradation.",
        ('hello', 'hi'): "Hello! How can I help you with your bearing analysis today?",
        ('bye', 'goodbye'): "Goodbye! Feel free to ask if you have more questions."
    }

    best_match_score = 0
    best_response = None

    for keywords, response in knowledge_base.items():
        score = len(tokens.intersection(keywords))
        if score > best_match_score:
            best_match_score = score
            best_response = response
    
    if best_match_score > 0:
        if best_response == "get_last_prediction":
            if 'last_rul_prediction' in st.session_state:
                res = st.session_state.last_rul_prediction
                return (f"The last prediction was for **Bearing {res['b']}** using the **{res['model']}** model.\n\n"
                        f"- **Predicted RUL:** {res['c']} cycles\n"
                        f"- **In Hours:** {res['h']:.2f} hours\n"
                        f"- **In Days:** {res['d']:.2f} days")
            else:
                return "I don't have a result yet. Please run a prediction in the 'Neural Networks' tab first."
        return best_response
    else:
        return "I'm not sure how to answer that. You can ask me about the last RUL result, or for definitions of terms like 'PCA', 'ARIMA', or 'stationarity'."

with tab6:
    st.header("Analysis Assistant")
    st.info("Ask me about the app's functionality or the latest RUL prediction results.")

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "How can I help you with your analysis?"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response = get_chatbot_response(prompt)
            st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
