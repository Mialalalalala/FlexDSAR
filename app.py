import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from retrieval_app import run_retrieval

def organize_frequencies(selected_freqs):
    """
    Ensures that the selected frequencies always follow the sequence:
    ["290 MHz P", "430 MHz P", "L"], keeping only the ones selected by the user.
    
    :param selected_freqs: List of frequencies selected by the user.
    :return: Ordered list of selected frequencies.
    """
    standard_order = ["290 MHz P", "430 MHz P", "L"]
    return [freq for freq in standard_order if freq in selected_freqs]

def organize_angles_dict(angle_dict):
    """
    Ensures that the angles for each frequency in the dictionary follow the sequence [30, 45, 60].

    :param angle_dict: Dictionary with frequencies as keys and lists of angles as values.
    :return: Dictionary with sorted angle lists.
    """
    standard_order = [30, 45, 60]
    return {freq: sorted(angles, key=lambda x: standard_order.index(x)) for freq, angles in angle_dict.items()}


st.title("🛰️📡🌍 Soil Moisture Profile & Vegetation Water Content Retrieval 🌱🌲🌳")

plt.rcParams.update({'font.size': 12})

# Select Land Cover Type
landcover = st.selectbox("Select Land Cover Type:", ["Grassland", "Shrub", "Deciduous", "Evergreen"])

# Select Frequency Bands
freq_options = ["290 MHz P", "430 MHz P", "L"]
selected_freqs = st.multiselect("Select Frequency Bands:", freq_options)
selected_freqs = organize_frequencies(selected_freqs)

# Select Polarization for each Frequency
pol_options = ["HH", "VV", "HH/HV", "VV/HV", "HH/VV/HV"]
selected_pols = {freq: st.selectbox(f"Select Polarization for {freq}:", pol_options) for freq in selected_freqs}

# Select Incidence Angles for each Frequency
angle_options = [30, 45, 60]
selected_angles = {freq: st.multiselect(f"Select Incidence Angles for {freq}:", angle_options) for freq in selected_freqs}
selected_angles = organize_angles_dict(selected_angles)

# Calibration Uncertainty Input
noise = st.number_input("Calibration Uncertainty (dB):", value=0.1)

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

# Display previously added comparison cases as a table
if st.session_state.comparison_cases:
    st.write("### 📋 Added Comparison Cases:")
    
    # Convert cases into a structured dataframe
    case_data = []
    for i, case in enumerate(st.session_state.comparison_cases):
        freqs = ", ".join(case["frequencies"])
        pols = ", ".join([f"{freq}: {case['polarizations'][freq]}" for freq in case["polarizations"]])
        angles = ", ".join([f"{freq}: " + ", ".join(map(str, case["angles"][freq])) + "°" for freq in case["angles"]])
        noise = f"{case['noise']} dB"
        rmse_values = [f"{val:.3f}" for val in case["rmse"]]  # Format RMSE values
        
        case_data.append([
            i+1, case["landcover"], freqs, pols, angles, noise, *rmse_values
        ])
    
    # Define column names dynamically
    depth_levels = ["SM at 0cm [m³/m³]", "SM at 20cm [m³/m³]", "SM at 50cm[m³/m³]", "Overall SM[m³/m³]","VWC[kg/m²]"]
    columns = ["Case #", "Land Cover", "Frequencies", "Polarizations", "Angles", "Noise"] + depth_levels
    
    # Create DataFrame and display
    df_cases = pd.DataFrame(case_data, columns=columns)
    st.dataframe(df_cases)
else:
    st.write("No cases have been added yet.")

# Button to finalize comparison and plot results
if st.button("Run Comparison"):
    if not st.session_state.comparison_cases:
        st.warning("No cases to compare. Please add at least one case.")
    else:
        st.write("### 🔍 Comparison of Retrieved RMSEs Across Cases:")
        depth_levels = ["SM at 0cm", "SM at 20cm", #"SM at 20cm","SM at 30cm",
                        "SM at 50cm",'Overall SM']
        
        fig, ax1 = plt.subplots(figsize=(10, 5))
        colors = ['blue', 'green', 'purple','black'] #'pink','deepskyblue'
        markers = ['o','o', 'o', 'o', '*']
        
        rmse = []
        for i, depth in enumerate(depth_levels):
            rmse_values = [case["rmse"][i] for case in st.session_state.comparison_cases]
            rmse.append(rmse_values)
            ax1.scatter(range(len(rmse_values)), rmse_values, color=colors[i], marker=markers[i], label=depth, s=100, facecolors='none')
        
        ax1.set_ylim(0, np.max(rmse)+0.02)
        ax1.axhline(y=0.075, color='red', linestyle='dashed')
        ax1.set_xticks(range(len(st.session_state.comparison_cases)))

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
        ax1.legend(bbox_to_anchor=(1.3, 1.1), loc='upper right')
        
        # Add VWC RMSE plot
        ax2 = ax1.twinx()
        vwc_rmse_values = [case["rmse"][4] for case in st.session_state.comparison_cases]
        ax2.scatter(range(len(vwc_rmse_values)), vwc_rmse_values, label="VWC", color='red', marker='^', s=100, facecolors='none')
        vwc_max_value = max(case["rmse"][4] for case in st.session_state.comparison_cases)
        ax2.set_ylim(0, vwc_max_value+0.05)
        ax2.set_ylabel("VWC RMSE (kg/m²)")
        ax2.legend(bbox_to_anchor=(1.27, 0.75), loc='upper right')
        plt.grid()
        st.pyplot(fig)
        
        # Clear comparison cases after plotting
        st.session_state.comparison_cases.clear()
