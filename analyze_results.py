# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import ast # Keep for safety, although not used for config parsing now
import numpy as np # For checking numeric types robustly
import matplotlib.ticker as mticker # For formatting axes

# --- Configuration ---
ACCURACY_CSV_PATH = 'linear_probing_accuracy_results.csv'
LOSS_CSV_PATH = 'simclr_loss_results.csv'
OUTPUT_DIR = 'analysis_plots'
TARGET_SUBSET_FRACTION_SET1 = 0.25
TARGET_AUGMENTATION_SET2 = 'all'
# Define the desired order for augmentation strategies in plots
AUGMENTATION_ORDER = [
    'baseline', 'jitter', 'blur', 'gray',
    'rotate', 'solarize', 'erase', 'all'
]
# *** Define ACTUAL column names from your CSV ***
ACC_COL_RUN_ID = 'run_id'
ACC_COL_CONFIG = 'augmentation_config' # Column containing the strategy name directly
ACC_COL_SUBSET = 'subset_fraction'
ACC_COL_TOP1 = 'top1_accuracy'
ACC_COL_TOP5 = 'top5_accuracy'
ACC_COL_PRETRAIN_TIME = 'total_pretrain_time_seconds'
ACC_COL_EVAL_TIME = 'eval_time_seconds'
LOSS_COL_RUN_ID = 'run_id'
LOSS_COL_CONFIG = 'augmentation_config' # Column containing the strategy name directly
LOSS_COL_SUBSET = 'subset_fraction'
LOSS_COL_EPOCH = 'epoch'
LOSS_COL_LOSS = 'loss'

# --- Create Output Directory ---
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Set Plotting Style ---
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 7)
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['legend.title_fontsize'] = 11

# --- Load Data ---
print("--- Loading Data ---")
try:
    df_acc = pd.read_csv(ACCURACY_CSV_PATH)
    df_loss = pd.read_csv(LOSS_CSV_PATH)
    print(f"Successfully loaded: '{ACCURACY_CSV_PATH}' ({len(df_acc)} rows)")
    print(f"Successfully loaded: '{LOSS_CSV_PATH}' ({len(df_loss)} rows)")

    # Strip whitespace from column headers
    df_acc.columns = df_acc.columns.str.strip()
    df_loss.columns = df_loss.columns.str.strip()
    print("Stripped whitespace from column headers.")
    print("Accuracy columns:", df_acc.columns.tolist())
    print("Loss columns:", df_loss.columns.tolist())

except FileNotFoundError as e:
    print(f"Error loading CSV file: {e}")
    exit()
except Exception as e:
    print(f"An unexpected error occurred while loading CSV files: {e}")
    exit()

# --- Data Preprocessing ---
print("\n--- Preprocessing Data ---")

# == Accuracy Dataframe ==
try:
    # Check if essential columns exist before proceeding
    required_acc_cols = [ACC_COL_RUN_ID, ACC_COL_CONFIG, ACC_COL_SUBSET, ACC_COL_TOP1, ACC_COL_TOP5, ACC_COL_PRETRAIN_TIME]
    missing_acc_cols = [col for col in required_acc_cols if col not in df_acc.columns]
    if missing_acc_cols:
        print(f"Error: Missing required columns in accuracy CSV: {missing_acc_cols}")
        print("Actual columns found:", df_acc.columns.tolist())
        exit()

    # Convert types using the correct column names
    df_acc[ACC_COL_TOP1] = pd.to_numeric(df_acc[ACC_COL_TOP1], errors='coerce')
    df_acc[ACC_COL_TOP5] = pd.to_numeric(df_acc[ACC_COL_TOP5], errors='coerce')
    df_acc[ACC_COL_SUBSET] = pd.to_numeric(df_acc[ACC_COL_SUBSET], errors='coerce')

    # Convert pre-train time to minutes
    df_acc['pre_train_time_minutes'] = pd.to_numeric(df_acc[ACC_COL_PRETRAIN_TIME], errors='coerce') / 60.0

except KeyError as e:
     print(f"Error accessing column: {e}. This likely means the column name definition is wrong.")
     print("Columns found in accuracy CSV:", df_acc.columns.tolist())
     exit()
except Exception as e:
     print(f"An unexpected error occurred during accuracy data type conversion: {e}")
     exit()

# Directly use the config column value as the strategy
if ACC_COL_CONFIG in df_acc.columns:
     # Convert to string, lowercase, and strip whitespace for consistency
     df_acc['augmentation_strategy'] = df_acc[ACC_COL_CONFIG].astype(str).str.lower().str.strip()
     print(f"Assigned 'augmentation_strategy' directly from '{ACC_COL_CONFIG}' in accuracy data.")
else:
     print(f"Error: Column '{ACC_COL_CONFIG}' not found in accuracy results. Cannot determine augmentation strategies.")
     exit()

# Check for rows where essential data might be missing AFTER conversion/assignment
initial_rows_acc = len(df_acc)
df_acc.dropna(subset=[ACC_COL_TOP1, ACC_COL_TOP5, ACC_COL_SUBSET, 'augmentation_strategy', 'pre_train_time_minutes', ACC_COL_RUN_ID], inplace=True)
dropped_rows_acc = initial_rows_acc - len(df_acc)
if dropped_rows_acc > 0:
    print(f"Warning: Dropped {dropped_rows_acc} rows from accuracy data due to missing values in essential columns after preprocessing.")


# == Loss Dataframe ==
try:
    # Check essential columns
    required_loss_cols = [LOSS_COL_RUN_ID, LOSS_COL_SUBSET, LOSS_COL_EPOCH, LOSS_COL_LOSS]
    # Check if config column exists, needed for direct assignment
    has_loss_config_col = LOSS_COL_CONFIG in df_loss.columns
    if not has_loss_config_col:
         print(f"Warning: Column '{LOSS_COL_CONFIG}' not found in loss CSV. Will attempt mapping via run_id.")

    missing_loss_cols = [col for col in required_loss_cols if col not in df_loss.columns]
    if missing_loss_cols:
        print(f"Error: Missing required columns in loss CSV: {missing_loss_cols}")
        print("Actual columns found:", df_loss.columns.tolist())
        exit()

    # Convert types
    df_loss[LOSS_COL_EPOCH] = pd.to_numeric(df_loss[LOSS_COL_EPOCH], errors='coerce')
    df_loss[LOSS_COL_LOSS] = pd.to_numeric(df_loss[LOSS_COL_LOSS], errors='coerce')
    df_loss[LOSS_COL_SUBSET] = pd.to_numeric(df_loss[LOSS_COL_SUBSET], errors='coerce')
except KeyError as e:
     print(f"Error accessing column: {e} in loss CSV.")
     print("Columns found in loss CSV:", df_loss.columns.tolist())
     exit()
except Exception as e:
     print(f"An unexpected error occurred during loss data type conversion: {e}")
     exit()

# Directly use the config column value as the strategy
# (Or map using run_id if config column missing in loss file)
if has_loss_config_col:
     df_loss['augmentation_strategy'] = df_loss[LOSS_COL_CONFIG].astype(str).str.lower().str.strip()
     print(f"Assigned 'augmentation_strategy' directly from '{LOSS_COL_CONFIG}' in loss data.")
elif ACC_COL_RUN_ID in df_acc.columns and LOSS_COL_RUN_ID in df_loss.columns:
     if not df_acc.empty:
        run_id_to_aug = df_acc.set_index(ACC_COL_RUN_ID)['augmentation_strategy'].to_dict()
        df_loss['augmentation_strategy'] = df_loss[LOSS_COL_RUN_ID].map(run_id_to_aug)
        if df_loss['augmentation_strategy'].isnull().any():
            print("Warning: Some loss rows could not be mapped to an augmentation strategy using run_id (might be due to drops in accuracy data).")
        print("Inferred 'augmentation_strategy' for loss data using run_id mapping from accuracy data.")
     else:
        print("Warning: Cannot map run_id for loss data because accuracy data is empty.")
        df_loss['augmentation_strategy'] = 'unknown' # Assign default
else:
     print("Warning: Cannot determine augmentation strategy for loss data (config missing and no run_id map possible). Loss plots by augmentation may be incorrect.")
     df_loss['augmentation_strategy'] = 'unknown' # Assign default

# Check for rows missing essential data in loss df
initial_rows_loss = len(df_loss)
df_loss.dropna(subset=[LOSS_COL_EPOCH, LOSS_COL_LOSS, LOSS_COL_SUBSET, 'augmentation_strategy', LOSS_COL_RUN_ID], inplace=True)
dropped_rows_loss = initial_rows_loss - len(df_loss)
if dropped_rows_loss > 0:
    print(f"Warning: Dropped {dropped_rows_loss} rows from loss data due to missing values in essential columns after preprocessing.")


# Check final state
print("\nFound augmentation strategies in accuracy data:", df_acc['augmentation_strategy'].unique())
print("Found augmentation strategies in loss data:", df_loss['augmentation_strategy'].unique())
print(f"Accuracy data shape after preprocessing: {df_acc.shape}")
print(f"Loss data shape after preprocessing: {df_loss.shape}")

if df_acc.empty or df_loss.empty:
     print("\nError: Dataframes are empty after preprocessing. Cannot generate plots.")
     exit()


print("\n--- Starting Plot Generation ---")

# --- Set 1 Plots: Comparing Augmentations (Fixed Subset Fraction) ---
print(f"\nGenerating plots for Set 1 (Subset Fraction = {TARGET_SUBSET_FRACTION_SET1})...")
df_acc_set1 = df_acc[df_acc[ACC_COL_SUBSET] == TARGET_SUBSET_FRACTION_SET1].copy()
df_loss_set1 = df_loss[df_loss[LOSS_COL_SUBSET] == TARGET_SUBSET_FRACTION_SET1].copy()

if df_acc_set1.empty:
    print(f"Warning: No accuracy data found for subset fraction {TARGET_SUBSET_FRACTION_SET1} after preprocessing. Skipping Set 1 plots.")
else:
    print(f"Found {len(df_acc_set1)} accuracy records for Set 1.")
    present_strategies_set1 = [aug for aug in AUGMENTATION_ORDER if aug in df_acc_set1['augmentation_strategy'].unique()]
    if not present_strategies_set1:
        print("Error: No recognized augmentation strategies left for Set 1 plots.")
    else:
        if len(present_strategies_set1) != len(AUGMENTATION_ORDER):
            print(f"Note: Not all expected augmentations found for Set 1. Plotting: {present_strategies_set1}")

        df_acc_set1['augmentation_strategy'] = pd.Categorical(df_acc_set1['augmentation_strategy'], categories=present_strategies_set1, ordered=True)
        df_acc_set1 = df_acc_set1.sort_values('augmentation_strategy')

        # 1.1: Accuracy Bar Chart
        # *** CORRECTED: Removed semicolons, ensured proper indentation ***
        plt.figure(figsize=(15, 8))
        try:
            df_melt = df_acc_set1.melt(id_vars='augmentation_strategy', value_vars=[ACC_COL_TOP1, ACC_COL_TOP5], var_name='Accuracy Type', value_name='Accuracy (%)')
            df_melt['Accuracy Type'] = df_melt['Accuracy Type'].map({ACC_COL_TOP1: 'Top-1', ACC_COL_TOP5: 'Top-5'})
            ax = sns.barplot( data=df_melt, x='augmentation_strategy', y='Accuracy (%)', hue='Accuracy Type', palette='viridis', order=present_strategies_set1 )
            plt.title(f'SimCLR Linear Probing Accuracy vs. Augmentation Strategy\n(CIFAR-10, Subset Fraction {TARGET_SUBSET_FRACTION_SET1 * 100:.0f}%, 50 Epochs Pre-train)')
            plt.xlabel('Augmentation Strategy')
            plt.ylabel('Accuracy (%)')
            max_acc = df_melt['Accuracy (%)'].max()
            plt.ylim(0, max_acc * 1.15 if not pd.isna(max_acc) else 10)
            plt.xticks(rotation=45, ha='right')
            for container in ax.containers:
                if container:
                    ax.bar_label(container, fmt='%.1f%%', fontsize=10, padding=3)
            plt.legend(title='Accuracy Type')
            plt.tight_layout()
            plot_filename = os.path.join(OUTPUT_DIR, '1_accuracy_vs_augmentation.png')
            plt.savefig(plot_filename, dpi=300)
            print(f"Saved: {plot_filename}")
        except Exception as e:
            print(f"Error generating plot 1.1 (Accuracy vs Augmentation): {e}")
        finally:
            plt.close()

        # 1.2: Time Bar Chart
        # *** CORRECTED: Removed semicolons, ensured proper indentation ***
        plt.figure(figsize=(15, 8))
        try:
            ax = sns.barplot( data=df_acc_set1, x='augmentation_strategy', y='pre_train_time_minutes', palette='coolwarm', order=present_strategies_set1)
            plt.title(f'SimCLR Pre-training Time vs. Augmentation Strategy\n(CIFAR-10, Subset Fraction {TARGET_SUBSET_FRACTION_SET1 * 100:.0f}%, 50 Epochs)')
            plt.xlabel('Augmentation Strategy')
            plt.ylabel('Pre-training Time (minutes)')
            plt.xticks(rotation=45, ha='right')
            if ax.containers:
                max_time = df_acc_set1['pre_train_time_minutes'].max()
                plt.ylim(0, max_time * 1.1 if not pd.isna(max_time) else 10)
                ax.bar_label(ax.containers[0], fmt='%.1f min', fontsize=10, padding=3)
            plt.tight_layout()
            plot_filename = os.path.join(OUTPUT_DIR, '1_pretrain_time_vs_augmentation.png')
            plt.savefig(plot_filename, dpi=300)
            print(f"Saved: {plot_filename}")
        except Exception as e:
            print(f"Error generating plot 1.2 (Time vs Augmentation): {e}")
        finally:
            plt.close()

        # 1.3: Loss Curves
        # *** CORRECTED: Removed semicolons, ensured proper indentation ***
        if df_loss_set1.empty:
            print(f"Warning: No loss data found for subset fraction {TARGET_SUBSET_FRACTION_SET1} after preprocessing. Skipping loss curve plot for Set 1.")
        else:
            print(f"Found {len(df_loss_set1)} loss records for Set 1.")
            plt.figure(figsize=(14, 8))
            try:
                present_loss_strategies_set1 = [aug for aug in AUGMENTATION_ORDER if aug in df_loss_set1['augmentation_strategy'].unique()]
                if not present_loss_strategies_set1:
                    print("Warning: No loss data with recognized augmentation strategies for Set 1 plot.")
                else:
                    df_loss_set1['augmentation_strategy'] = pd.Categorical(df_loss_set1['augmentation_strategy'], categories=present_loss_strategies_set1, ordered=True)
                    df_loss_set1 = df_loss_set1.sort_values('augmentation_strategy')
                    sns.lineplot( data=df_loss_set1, x=LOSS_COL_EPOCH, y=LOSS_COL_LOSS, hue='augmentation_strategy', palette='tab10', hue_order=present_loss_strategies_set1, legend='full', errorbar=None )
                    plt.title(f'SimCLR Pre-training Loss Curves by Augmentation Strategy\n(CIFAR-10, Subset Fraction {TARGET_SUBSET_FRACTION_SET1 * 100:.0f}%)')
                    plt.xlabel('Pre-training Epoch')
                    plt.ylabel('Contrastive Loss')
                    plt.legend(title='Augmentation Strategy', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
                    plt.tight_layout(rect=[0, 0, 0.85, 1])
                    plot_filename = os.path.join(OUTPUT_DIR, '1_loss_curves_vs_augmentation.png')
                    plt.savefig(plot_filename, dpi=300)
                    print(f"Saved: {plot_filename}")
            except Exception as e:
                print(f"Error generating plot 1.3 (Loss vs Augmentation): {e}")
            finally:
                plt.close()

# --- Set 2 Plots: Comparing Subset Fractions (Fixed 'all' Augmentation) ---
print(f"\nGenerating plots for Set 2 (Augmentation Strategy = '{TARGET_AUGMENTATION_SET2}')...")
df_acc_target_aug = df_acc[df_acc['augmentation_strategy'] == TARGET_AUGMENTATION_SET2].copy()
df_loss_target_aug = df_loss[df_loss['augmentation_strategy'] == TARGET_AUGMENTATION_SET2].copy()
present_subsets_set2 = sorted(df_acc_target_aug[ACC_COL_SUBSET].unique())

if df_acc_target_aug.empty:
    print(f"Warning: No accuracy data found for augmentation strategy '{TARGET_AUGMENTATION_SET2}' after preprocessing. Skipping Set 2 plots.")
else:
    print(f"Found {len(df_acc_target_aug)} accuracy records for Set 2 across subsets: {present_subsets_set2}")
    df_acc_set2 = df_acc_target_aug.sort_values(ACC_COL_SUBSET)

    # 2.1: Accuracy Line Plot
    # *** CORRECTED: Removed semicolons, ensured proper indentation ***
    plt.figure(figsize=(12, 7))
    try:
        df_melt_set2 = df_acc_set2.melt( id_vars=ACC_COL_SUBSET, value_vars=[ACC_COL_TOP1, ACC_COL_TOP5], var_name='Accuracy Type', value_name='Accuracy (%)' )
        df_melt_set2['Accuracy Type'] = df_melt_set2['Accuracy Type'].map({ACC_COL_TOP1: 'Top-1', ACC_COL_TOP5: 'Top-5'})
        sns.lineplot( data=df_melt_set2, x=ACC_COL_SUBSET, y='Accuracy (%)', hue='Accuracy Type', style='Accuracy Type', markers=True, markersize=8, palette='magma')
        plt.title(f'SimCLR Linear Probing Accuracy vs. Pre-training Subset Fraction\n(CIFAR-10, \'{TARGET_AUGMENTATION_SET2}\' Augmentations, 50 Epochs Pre-train)')
        plt.xlabel('Pre-training Subset Fraction')
        plt.ylabel('Accuracy (%)')
        plt.xticks(present_subsets_set2)
        max_acc_s2 = df_melt_set2['Accuracy (%)'].max()
        plt.ylim(0, max_acc_s2 * 1.1 if not pd.isna(max_acc_s2) else 10)
        plt.gca().xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        plt.legend(title='Accuracy Type')
        plt.tight_layout()
        plot_filename = os.path.join(OUTPUT_DIR, '2_accuracy_vs_subset_fraction.png')
        plt.savefig(plot_filename, dpi=300)
        print(f"Saved: {plot_filename}")
    except Exception as e:
        print(f"Error generating plot 2.1 (Accuracy vs Subset): {e}")
    finally:
        plt.close()

    # 2.2: Time Line Plot
    # *** CORRECTED: Removed semicolons, ensured proper indentation ***
    plt.figure(figsize=(12, 7))
    try:
        sns.lineplot( data=df_acc_set2, x=ACC_COL_SUBSET, y='pre_train_time_minutes', marker='o', markersize=8, color='teal' )
        plt.title(f'SimCLR Pre-training Time vs. Subset Fraction\n(CIFAR-10, \'{TARGET_AUGMENTATION_SET2}\' Augmentations, 50 Epochs)')
        plt.xlabel('Pre-training Subset Fraction')
        plt.ylabel('Pre-training Time (minutes)')
        plt.xticks(present_subsets_set2)
        max_time_s2 = df_acc_set2['pre_train_time_minutes'].max()
        plt.ylim(0, max_time_s2 * 1.1 if not pd.isna(max_time_s2) else 10)
        plt.gca().xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        plt.tight_layout()
        plot_filename = os.path.join(OUTPUT_DIR, '2_pretrain_time_vs_subset_fraction.png')
        plt.savefig(plot_filename, dpi=300)
        print(f"Saved: {plot_filename}")
    except Exception as e:
        print(f"Error generating plot 2.2 (Time vs Subset): {e}")
    finally:
        plt.close()

    # 2.3: Loss Curves
    # *** CORRECTED: Removed semicolons, ensured proper indentation ***
    df_loss_set2 = df_loss_target_aug.sort_values(LOSS_COL_SUBSET)
    if df_loss_set2.empty:
        print(f"Warning: No loss data found for augmentation strategy '{TARGET_AUGMENTATION_SET2}' after preprocessing. Skipping loss curve plot for Set 2.")
    else:
        print(f"Found {len(df_loss_set2)} loss records for Set 2.")
        plt.figure(figsize=(14, 8))
        try:
            df_loss_set2['subset_fraction_str'] = (df_loss_set2[LOSS_COL_SUBSET] * 100).astype(int).astype(str) + '%'
            sns.lineplot( data=df_loss_set2, x=LOSS_COL_EPOCH, y=LOSS_COL_LOSS, hue='subset_fraction_str', palette='crest', hue_order=sorted(df_loss_set2['subset_fraction_str'].unique(), key=lambda x: float(x.strip('%'))), legend='full', errorbar=None )
            plt.title(f'SimCLR Pre-training Loss Curves by Subset Fraction\n(CIFAR-10, \'{TARGET_AUGMENTATION_SET2}\' Augmentations)')
            plt.xlabel('Pre-training Epoch')
            plt.ylabel('Contrastive Loss')
            plt.legend(title='Subset Fraction', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            plt.tight_layout(rect=[0, 0, 0.85, 1])
            plot_filename = os.path.join(OUTPUT_DIR, '2_loss_curves_vs_subset_fraction.png')
            plt.savefig(plot_filename, dpi=300)
            print(f"Saved: {plot_filename}")
        except Exception as e:
            print(f"Error generating plot 2.3 (Loss vs Subset): {e}")
        finally:
            plt.close()

print(f"\n--- Plot generation finished! Check the '{OUTPUT_DIR}' directory. ---")