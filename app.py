import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from retrieval import run_retrieval

st.title("🌍 Soil Moisture Profile & Vegetation Water Content Retrieval")

# Select Land Cover Type
landcover = st.selectbox("Select Land Cover Type:", ["Grassland", "Shrub", "Deciduous", "Evergreen"])

# Select Frequency Bands
freq_options = ["290 MHz P", "430 MHz P", "L"]
selected_freqs = st.multiselect("Select Frequency Bands:", freq_options)

# Select Polarization for each Frequency
pol_options = ["HH/HV", "VV/HV", "HH", "VV", "HH/VV/HV"]
selected_pols = {freq: st.selectbox(f"Select Polarization for {freq}:", pol_options) for freq in selected_freqs}

# Select Incidence Angles for each Frequency
angle_options = [30, 45, 60]
selected_angles = {freq: st.multiselect(f"Select Incidence Angles for {freq}:", angle_options) for freq in selected_freqs}

# Calibration Uncertainty Input
noise = st.number_input("Calibration Uncertainty (dB):", value=0.1, step=0.01)

# Store comparison cases
if "comparison_cases" not in st.session_state:
    st.session_state.comparison_cases = []

# Add case to comparison
if st.button("Add to Comparison"):
    case = {
        "landcover": landcover,
        "frequencies": selected_freqs,
        "polarizations": selected_pols,
        "angles": selected_angles,
        "noise": noise
    }
    rmse = run_retrieval(case)
    case["rmse"] = rmse
    st.session_state.comparison_cases.append(case)
    st.success("Case added to comparison!")

# Button to finalize comparison and plot results
if st.button("Done Comparison"):
    if not st.session_state.comparison_cases:
        st.warning("No cases to compare. Please add at least one case.")
    else:
        st.write("### 🔍 Comparison of Retrieved RMSEs Across Cases:")
        depth_levels = ["SM at 10cm", "SM at 20cm", "SM at 50cm"]
        
        fig, ax1 = plt.subplots(figsize=(10, 5))
        colors = ['orange', 'green', 'purple']
        
        for i, depth in enumerate(depth_levels):
            rmse_values = [case["rmse"][i] for case in st.session_state.comparison_cases]
            ax1.scatter(range(len(rmse_values)), rmse_values, color=colors[i], label=depth, s=100, facecolors='none')
        ax1.set_ylim(0, np.max(rmse_values)+0.02)
        ax1.axhline(y=0.075, color='red', linestyle='dashed')
        ax1.set_xticks(range(len(st.session_state.comparison_cases)))
#         print(st.session_state.comparison_cases)
        case_labels = []
        for case in st.session_state.comparison_cases:
            freqs = ", ".join(case["frequencies"])
            pols = ", ".join([f"{freq}: {case['polarizations'][freq]}" for freq in case["polarizations"]])
            angles = ", ".join([f"{freq}: " + ", ".join(map(str, case["angles"][freq])) + "°" for freq in case["angles"]])
            noise = f"Noise: {case['noise']}"
            case_labels.append(f"{freqs} | {pols} | {angles} | {noise}")

        
        ax1.set_xticklabels(case_labels, rotation=45, ha='right')
        ax1.set_xlabel("Cases")

        ax1.set_ylabel("SM RMSE (m³/m³)")
        ax1.legend()
        
        # Add VWC RMSE plot
        ax2 = ax1.twinx()
        vwc_rmse_values = [case["rmse"][3] for case in st.session_state.comparison_cases]
        ax2.scatter(range(len(vwc_rmse_values)), vwc_rmse_values, label="VWC", color='red', marker='^', s=100, facecolors='none')
        vwc_max_value = max(case["rmse"][3] for case in st.session_state.comparison_cases)
        ax2.set_ylim(0, vwc_max_value+0.05)
        ax2.set_ylabel("VWC RMSE (kg/m²)")
        ax2.legend(loc='lower right')
        plt.grid()
        st.pyplot(fig)
        
        # Clear comparison cases after plotting
        st.session_state.comparison_cases.clear()
