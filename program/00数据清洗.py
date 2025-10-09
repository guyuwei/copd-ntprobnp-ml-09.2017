import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from scipy.stats import mannwhitneyu, kruskal, chi2_contingency, fisher_exact
import os
from collections import OrderedDict

# ==================== Configuration ====================
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

input_dir = "/Users/gyw/Desktop/Project/2025/NT-proBNP～COPD/Resources/Raw/"
output_dir = "/Users/gyw/Desktop/Project/2025/NT-proBNP～COPD/00dataset/"
os.makedirs(output_dir, exist_ok=True)

# Column definitions
cardiac_cols = ['AF', 'LBBB', 'PAC', 'PVC', 'RBBB', 'SVT']
bool_index = "CAD,DM,GOLD(1-4),GENDER,HF,HTN,OUTCOMES,TREATMENT".split(",")
int_index = "ALT,AST,AGE,OI,PASP,PCO2,RV/TLC%,DAYSOFHOSPITALIZATION,SMOKINGINDEX".split(",")
float_index = "BMI,CL,CR,D-D,FEV1-BEST,FVCPRED,K,MEDICALEXPENSES,NT-PROBNP,NA,PH,HS-CTN,DLCO%PRED,FEV1%PRED,FEV1/FVC%,FVC%PRED".split(',')
time_index = "ADMISSIONTIME,DISCHARGETIME".split(",")
AF_index = "AF,LBBB,PAC,PVC,RBBB,SVT".split(",")


# ==================== Data Processing ====================
def preprocess_data(df, is_arrhythmia=False):
    """Enhanced preprocessing with count tracking"""
    print(f"\nPreprocessing {'arrhythmia' if is_arrhythmia else 'normal'} dataset (Initial count: {len(df)})")

    # Standardize column names
    df.columns = [col.upper() for col in df.columns]

    # Initialize cardiac columns
    for col in cardiac_cols:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        df[col] = np.where(df[col] > 0, 1, 0)

    if not is_arrhythmia:
        df[cardiac_cols] = 0

    df['CAD'] = df[cardiac_cols].max(axis=1).astype(int)
    print(f"CAD positive cases: {df['CAD'].sum()} ({df['CAD'].mean():.1%})")

    # Type conversion
    numeric_cols = int_index + float_index
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    categorical_cols = ['GENDER', 'HTN', 'DM']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')

    print(f"Final preprocessed count: {len(df)}")
    return df


# ==================== Data Loading ====================
print("\nLoading data...")
try:
    df_normal = pd.read_excel(os.path.join(input_dir, "COPD无心律失常04.30_2.xlsx"))
    df_arrhythmia = pd.read_excel(os.path.join(input_dir, "COPD合并心律失常04.30_2.xlsx"))

    df_normal = preprocess_data(df_normal, is_arrhythmia=False)
    df_arrhythmia = preprocess_data(df_arrhythmia, is_arrhythmia=True)

    print("\nColumn alignment check:")
    print(f"Normal group columns: {len(df_normal.columns)}")
    print(f"Arrhythmia group columns: {len(df_arrhythmia.columns)}")
    print(f"Common columns: {len(set(df_normal.columns) & set(df_arrhythmia.columns))}")

except Exception as e:
    print(f"Data loading failed: {str(e)}")
    exit()

# ==================== Data Merging ====================
print("\nMerging datasets...")
try:
    all_columns = list(set(df_normal.columns) | set(df_arrhythmia.columns))
    df_normal = df_normal.reindex(columns=all_columns, fill_value=0)
    df_arrhythmia = df_arrhythmia.reindex(columns=all_columns, fill_value=0)

    combined_df = pd.concat([df_normal, df_arrhythmia], axis=0, ignore_index=True)
    print(f"Merged successfully! Total samples: {len(combined_df)}")
    print(f"CAD prevalence in combined data: {combined_df['CAD'].sum()} ({combined_df['CAD'].mean():.1%})")

except Exception as e:
    print(f"Merge failed: {str(e)}")
    exit()

# ==================== Missing Value Handling ====================
print("\nHandling missing values...")
try:
    numeric_cols = combined_df.select_dtypes(include=[np.number]).columns.difference(cardiac_cols + ['CAD'])

    # Track missing values
    missing_before = combined_df[numeric_cols].isna().sum().sum()
    print(f"Missing values before imputation: {missing_before}")

    imputer = IterativeImputer(max_iter=20, random_state=42)
    combined_df[numeric_cols] = imputer.fit_transform(combined_df[numeric_cols])

    missing_after = combined_df[numeric_cols].isna().sum().sum()
    print(f"Missing values after imputation: {missing_after}")

    if missing_after > 0:
        raise ValueError(f"Imputation failed for {missing_after} values")

except Exception as e:
    print(f"Missing value handling failed: {str(e)}")
    exit()


# ==================== Enhanced Subgroup Analysis ====================
def subgroup_analysis(data, group_col, group_name, output_dir):
    """Enhanced subgroup analysis with detailed counting"""
    print(f"\nRunning subgroup analysis for {group_name} (grouping by {group_col})")

    # Validate group column
    if group_col not in data.columns:
        print(f"Warning: {group_col} not found in dataset")
        return None

    groups = data[group_col].dropna().unique()
    groups.sort()
    print(f"Detected groups: {groups} with counts: {data[group_col].value_counts().to_dict()}")

    # Prepare analysis variables
    analysis_vars = {
        'continuous': float_index + int_index,
        'categorical': bool_index
    }

    results = []

    # Continuous variables analysis - Modified to output median and IQR in one column
    for var in analysis_vars['continuous']:
        if var not in data.columns:
            continue

        row = {'Variable': var, 'Type': 'Continuous'}
        valid_counts = []

        for group in groups:
            group_data = data[data[group_col] == group][var].dropna()
            count = len(group_data)
            valid_counts.append(count)

            # Calculate median and IQR
            median = round(group_data.median(), 2)
            q1 = round(group_data.quantile(0.25), 2)
            q3 = round(group_data.quantile(0.75), 2)

            # Store in combined format
            row[f'{group}'] = f"{median} ({q1}-{q3})"
            row[f'{group}_N'] = count

        # Statistical testing
        if all(c > 5 for c in valid_counts) and len(groups) > 1:
            group_data = [data[data[group_col] == g][var].dropna() for g in groups]
            try:
                if len(groups) == 2:
                    _, p = mannwhitneyu(*group_data)
                else:
                    _, p = kruskal(*group_data)
                row['p-value'] = f"{p:.4f}"
                row['Significant'] = p < 0.05
            except:
                row['p-value'] = 'NA'
                row['Significant'] = False
        else:
            row['p-value'] = 'NA (insufficient data)'
            row['Significant'] = False

        results.append(row)

    # Categorical variables analysis (unchanged)
    for var in analysis_vars['categorical']:
        if var not in data.columns:
            continue

        row = {'Variable': var, 'Type': 'Categorical'}
        cross_tab = pd.crosstab(data[var], data[group_col])

        for group in groups:
            total = cross_tab[group].sum()
            row[f'{group}_N'] = total
            for val in cross_tab.index:
                count = cross_tab.loc[val, group]
                row[f'{group}_{val}'] = f"{count} ({count / total:.1%})"

        # Statistical testing
        try:
            if cross_tab.size >= 4:
                if cross_tab.min().min() > 5:
                    _, p, _, _ = chi2_contingency(cross_tab)
                else:
                    _, p = fisher_exact(cross_tab)
                row['p-value'] = f"{p:.4f}"
                row['Significant'] = p < 0.05
            else:
                row['p-value'] = 'NA (small sample)'
                row['Significant'] = False
        except Exception as e:
            row['p-value'] = f'NA ({str(e)})'
            row['Significant'] = False

        results.append(row)

    # Save results
    result_df = pd.DataFrame(results)
    output_path = os.path.join(output_dir, f"00_SubgroupAnalysis_{group_name}.csv")
    result_df.to_csv(output_path, index=False)

    # Print summary
    num_significant = sum(r.get('Significant', False) for r in results)
    print(f"Analysis complete. Found {num_significant} significant associations.")
    print(f"Results saved to {output_path}")

    return result_df


# ==================== Execute Analyses ====================
print("\nStarting comprehensive subgroup analyses...")

# Save full dataset with counts
combined_df.to_csv(os.path.join(output_dir, "00_AllData.csv"), index=False)
print(f"\nSaved full dataset with {len(combined_df)} records")

# Cardiac subgroup analyses
for disease in cardiac_cols:
    subgroup = combined_df[combined_df[disease] == 1]
    count = len(subgroup)
    if count > 0:
        print(f"\nAnalyzing {disease} subgroup ({count} cases, {count / len(combined_df):.1%} of total)")
        subgroup.to_csv(os.path.join(output_dir, f"00_{disease}.csv"), index=False)
        subgroup_analysis(combined_df, disease, f"Disease_{disease}", output_dir)
    else:
        print(f"Skipping {disease} - no cases found")

# BBB composite analysis
combined_df['BBB'] = ((combined_df['RBBB'] == 1) | (combined_df['LBBB'] == 1)).astype(int)
bbb_count = combined_df['BBB'].sum()
if bbb_count > 0:
    print(f"\nAnalyzing BBB composite ({bbb_count} cases)")
    combined_df[combined_df['BBB'] == 1].to_csv(os.path.join(output_dir, "00_BBB.csv"), index=False)
    subgroup_analysis(combined_df, 'BBB', "BBB", output_dir)

# Healthy controls
combined_df['Healthy'] = (combined_df[cardiac_cols].sum(axis=1) == 0).astype(int)
healthy_count = combined_df['Healthy'].sum()
print(f"\nAnalyzing healthy controls ({healthy_count} cases)")
combined_df[combined_df['Healthy'] == 1].to_csv(os.path.join(output_dir, "00_Healthy.csv"), index=False)
subgroup_analysis(combined_df, 'Healthy', "Healthy", output_dir)

# Core analyses
print("\nRunning core analyses:")
subgroup_analysis(combined_df, 'CAD', "CAD", output_dir)

if 'GENDER' in combined_df.columns:
    gender_counts = combined_df['GENDER'].value_counts()
    print(f"\nGender distribution: {gender_counts.to_dict()}")
    subgroup_analysis(combined_df, 'GENDER', "Gender", output_dir)

# ==================== Final Report ====================
print("\nANALYSIS COMPLETE")
print("=" * 50)
print(f"Total patients analyzed: {len(combined_df)}")
print(f"CAD prevalence: {combined_df['CAD'].sum()} ({combined_df['CAD'].mean():.1%})")
print("\nGenerated files:")
for f in os.listdir(output_dir):
    if f.startswith('00_'):
        size = os.path.getsize(os.path.join(output_dir, f)) / 1024
        print(f"- {f} ({size:.1f} KB)")