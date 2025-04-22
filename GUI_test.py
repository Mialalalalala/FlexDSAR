import os
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt
import numpy as np
from retrieval import run_retrieval
from sklearn.metrics import mean_squared_error

plt.rcParams.update({'font.size': 16})

# Store comparison cases
comparison_cases = []

# Function to update UI based on frequency selection
def update_ui():
    for freq in freq_vars:
        state = tk.NORMAL if freq_vars[freq].get() else tk.DISABLED
        pol_menus[freq].config(state=state)
        for checkbox in incidence_angle_menus[freq]:
            checkbox.config(state=state)

# Function to add current selection to comparison cases
def add_to_comparison():
    try:
        noise = float(entry_noise.get())
        landcover = landcover_var.get()
        selected_frequencies = [freq for freq in freq_vars if freq_vars[freq].get()]
        
        if not selected_frequencies:
            raise ValueError("Please select at least one frequency band.")

        selected_angles = {freq: [angle for angle, var in incidence_angle_vars[freq].items() if var.get()] for freq in selected_frequencies}

        case = {
            "landcover": landcover,
            "frequencies": selected_frequencies,
            "polarizations": {freq: pol_vars[freq].get() for freq in selected_frequencies},
            "angles": {freq:selected_angles[freq] for freq in selected_frequencies},
            "noise": noise,
            "rmse": np.nan
        }

        rmse = run_retrieval(case)
        case["rmse"]=rmse
        comparison_cases.append(case)

       # Update Treeview Table
        case_number = len(comparison_cases)
        row_data = (
            f"Case {case_number}",
            landcover,
            ", ".join(selected_frequencies),
            ", ".join([f"{f}: {pol_vars[f].get()}" for f in selected_frequencies]),
            ", ".join([f"{f}: {selected_angles[f]}" for f in selected_frequencies]),
            f"{noise:.2f}",
            ", ".join([f"{v:.3f}" for v in rmse])
        )
        tree.insert("", "end", values=row_data)

        messagebox.showinfo("Comparison", f"Case added.")
    except ValueError as e:
        messagebox.showerror("Input Error", str(e))

# Function to compute and plot RMSE summary
def plot_comparison_summary():
    if not comparison_cases:
        messagebox.showerror("Error", "No cases to compare. Please add cases first.")
        return
    
    fig, ax1 = plt.subplots(figsize=(16, 8))
    depth_levels = ["SM at 0cm", "SM at 20cm", "SM at 50cm",'Overall SM']
    colors = ['blue', 'green', 'purple','black'] #'pink','deepskyblue'
    markers = ['o','o', 'o', '*']
    
    rmse = []
    for i, depth_label in enumerate(depth_levels):
        rmse_values = [case["rmse"][i] for case in comparison_cases]
        rmse.append(rmse_values)
        ax1.scatter(range(len(comparison_cases)), rmse_values, label=depth_label, color=colors[i], marker=markers[i],s=200, linewidths=2,facecolors='none')
    ax1.set_ylim(0, 0.2)#np.max(rmse)+0.02)
    # ax1.axhline(y=0.05, color='red', linestyle='dashed',)
    ax1.legend(loc='lower right',framealpha=0.5)

    ax2 = ax1.twinx()
    vwc_rmse_values = [case["rmse"][4] for case in comparison_cases]
    ax2.scatter(range(len(comparison_cases)), vwc_rmse_values, label="VWC", color='red', marker='^',s=200, linewidths=2,facecolors='none')
    
    
    # case_labels = [" + ".join([ # Convert list to a comma-separated string
    # ", ".join([f"{freq}: {case['polarizations'][freq]}" for freq in case['polarizations']]),  # Format polarization info
    # ", ".join([f"{freq}: {case['angles'][freq]}°" for freq in case['angles']]),  # Format incidence angles
    # str(case['noise'])  # Convert noise value to string
    # ]) for case in comparison_cases]
    
    # ax1.set_xticks(range(len(comparison_cases)))
    # ax1.set_xticklabels(case_labels, rotation=45, ha='right')

    case_labels = [f"Case {i+1}" for i in range(len(comparison_cases))]
    ax1.set_xticks(range(len(comparison_cases)))
    ax1.set_xticklabels(case_labels)

    ax1.set_xlabel("Cases")
    ax1.set_ylabel("SM RMSE (m³/m³)")
    ax1.set_title("Comparison of Retrieved RMSE of SM and VWC Across Cases")

    ax2.set_ylabel("VWC RMSE (kg/m²)")
    ax2.legend(loc='lower left',framealpha=0.5)
    vwc_max_value = max(case["rmse"][4] for case in comparison_cases)
    ax2.set_ylim(0, 6)#vwc_max_value+0.15)
    plt.grid()
    plt.tight_layout()
    plt.show()
    
    comparison_cases.clear()
    tree.delete(*tree.get_children())


def download_table():
    if not tree.get_children():
        messagebox.showerror("Error", "No data to download.")
        return

    data = []
    for child in tree.get_children():
        row = tree.item(child)["values"]
        data.append(row)

    columns = ["Case #", "Land Cover", "Frequencies", "Polarizations", "Angles", "Noise (dB)", "RMSE (SM@0,20,50,All,VWC)"]
    df = pd.DataFrame(data, columns=columns)

    filename = filedialog.asksaveasfilename(
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv")],
        title="Save table as..."
    )
    if filename:
        df.to_csv(filename, index=False)
        messagebox.showinfo("Download Complete", f"Saved as '{filename}'")

# GUI Setup
root = tk.Tk()
root.title("Soil Moisture Profile and Vegetation Water Content Retrieval")
# root.geometry("1200x600")

# Land Cover Selection Buttons
tk.Label(root, text="Land Cover Type:").grid(row=0, column=0)
landcover_var = tk.StringVar(root, "Grassland")
landcover_frame = tk.Frame(root)
landcover_frame.grid(row=0, column=1, columnspan=3)
for cover in ["Grassland", "Shrub", "Deciduous", "Evergreen"]:
    tk.Radiobutton(landcover_frame, text=cover, variable=landcover_var, value=cover).pack(side=tk.LEFT)

# Frequency Band Selection Checkboxes
tk.Label(root, text="Frequency Bands:").grid(row=1, column=0)
freq_vars = {}
freq_frame = tk.Frame(root)
freq_frame.grid(row=1, column=1, columnspan=3)
for freq in ["290 MHz P", "430 MHz P", "L"]:
    freq_vars[freq] = tk.BooleanVar()
    tk.Checkbutton(freq_frame, text=freq, variable=freq_vars[freq], command=update_ui).pack(side=tk.LEFT)

# Polarization and Incidence Angle Selection
pol_vars = {}
incidence_angle_vars = {}
pol_menus = {}
incidence_angle_menus = {}

for i, freq in enumerate(["290 MHz P", "430 MHz P", "L"]):
    tk.Label(root, text=f"{freq} Polarization:", font=(None, 12)).grid(row=2, column=i)
    pol_vars[freq] = tk.StringVar(root)
    pol_menus[freq] = ttk.Combobox(root, textvariable=pol_vars[freq], values=["HH/HV", "VV/VH", "HH/VV/HV/VH"], state=tk.DISABLED)
    pol_menus[freq].grid(row=3, column=i)

    tk.Label(root, text=f"{freq} Incidence Angles:", font=(None, 12)).grid(row=4, column=i)
    incidence_angle_vars[freq] = {angle: tk.BooleanVar() for angle in [30, 45, 60]}
    incidence_angle_menus[freq] = [tk.Checkbutton(root, text=str(angle), variable=incidence_angle_vars[freq][angle], state=tk.DISABLED) for angle in [30, 45, 60]]
    for j, checkbox in enumerate(incidence_angle_menus[freq]):
        checkbox.grid(row=5 + j, column=i)

# Noise Input
tk.Label(root, text="Calibration Uncertainty (dB):").grid(row=8, column=0)
entry_noise = tk.Entry(root)
entry_noise.grid(row=8, column=1, columnspan=2)

# Comparison Buttons
tk.Button(root, text="Add to Comparison", command=add_to_comparison).grid(row=9, column=0, columnspan=3)
tk.Button(root, text="Run Comparison", command=plot_comparison_summary).grid(row=10, column=0, columnspan=3)
tk.Button(root, text="Download Table", command=download_table).grid(row=11, column=0, columnspan=3)



# Table: Comparison Summary
columns = ("Case #","Land Cover", "Frequencies", "Polarizations", "Angles", "Noise (dB)", "RMSE (SM@0,20,50,All,VWC)")
tree = ttk.Treeview(root, columns=columns, show="headings", height=5)
for col in columns:
    tree.heading(col, text=col)
    tree.column(col, width=150, anchor='center')
tree.grid(row=12, column=0, columnspan=3, sticky='nsew')

scrollbar = ttk.Scrollbar(root, orient="vertical", command=tree.yview)
tree.configure(yscrollcommand=scrollbar.set)
scrollbar.grid(row=12, column=3, sticky='ns')

root.mainloop()