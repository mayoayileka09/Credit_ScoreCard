# ============================================================
# LendingClub PD Model (WoE/IV + Logistic Regression)
# Extracted & consolidated from pasted Kaggle notebook text
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score
)

from imblearn.over_sampling import RandomOverSampler

# Optional visualization dependency (used in the notebook)
try:
    from yellowbrick.target import ClassBalance
    HAS_YELLOWBRICK = True
except Exception:
    HAS_YELLOWBRICK = False


sns.set(context="notebook")
sns.set_style("whitegrid", {"axes.grid": False})


# -----------------------------
# Load data
# -----------------------------
CSV_PATH = r'/Users/user/Desktop/Research/Dr. Godwin/Credit Model/loan_data_2007_2014/loan_data_2007_2014.csv'
data = pd.read_csv(CSV_PATH)


# -----------------------------
# Target creation
# -----------------------------
data["bad_loan"] = np.where(
    data.loc[:, "loan_status"].isin([
        "Charged Off",
        "Default",
        "Late (31-120 days)",
        "Does not meet the credit policy. Status:Charged Off"
    ]),
    0,
    1
)

data.drop(columns=["loan_status"], inplace=True)
if "Unnamed: 0" in data.columns:
    data.drop("Unnamed: 0", inplace=True, axis=1)

X = data.drop("bad_loan", axis=1)
y = data["bad_loan"]


# -----------------------------
# Missingness inspection (from notebook)
# -----------------------------
missing_values = data.isnull().mean()
_ = missing_values[missing_values > 0.7]


# -----------------------------
# Drop irrelevant / future-leakage / high-missing columns
# -----------------------------
columns_to_drop = [
    "id", "member_id", "sub_grade", "emp_title", "url", "desc", "title", "zip_code", "next_pymnt_d",
    "recoveries", "collection_recovery_fee", "total_rec_prncp", "total_rec_late_fee", "desc",
    "mths_since_last_record", "mths_since_last_major_derog", "annual_inc_joint", "dti_joint",
    "verification_status_joint", "open_acc_6m", "open_il_6m", "open_il_12m", "open_il_24m",
    "mths_since_rcnt_il", "total_bal_il", "il_util", "open_rv_12m", "open_rv_24m", "max_bal_bc",
    "all_util", "inq_fi", "total_cu_tl", "inq_last_12m", "policy_code"
]
columns_to_drop = [c for c in columns_to_drop if c in data.columns]
data.drop(columns=columns_to_drop, inplace=True, axis=1)

data.dropna(inplace=True)


# -----------------------------
# Correlation heatmap (original notebook did this twice)
# -----------------------------
corr_df = data.select_dtypes(include=[np.number])

mask = np.zeros_like(corr_df.corr().fillna(0), dtype=bool)
mask[np.triu_indices_from(mask)] = True

plt.figure(figsize=(24, 24))
sns.heatmap(
    corr_df.corr(),
    mask=mask,
    annot=True,
    cmap="inferno",
    vmin=-1,
    fmt=".1g",
    edgecolor="w",
    linewidth=0.6
)
plt.tight_layout()
plt.show()


# -----------------------------
# Remove multicollinear features (as in notebook)
# -----------------------------
to_drop_multicollinear = [
    "loan_amnt", "revol_bal", "funded_amnt", "funded_amnt_inv", "installment",
    "total_pymnt_inv", "out_prncp_inv", "total_acc"
]
to_drop_multicollinear = [c for c in to_drop_multicollinear if c in data.columns]
data.drop(columns=to_drop_multicollinear, inplace=True)

corr_df = data.select_dtypes(include=[np.number])

mask = np.zeros_like(corr_df.corr().fillna(0), dtype=bool)
mask[np.triu_indices_from(mask)] = True

plt.figure(figsize=(24, 24))
sns.heatmap(
    corr_df.corr(),
    mask=mask,
    annot=True,
    cmap="inferno",
    vmin=-1,
    fmt=".1g",
    edgecolor="w",
    linewidth=0.6
)
plt.tight_layout()
plt.show()

# -----------------------------
# Data type transformations
# -----------------------------
def emp_length_convert(df, column):
    df[column] = df[column].str.replace(r"\+ years", "", regex=True)
    df[column] = df[column].str.replace(r"< 1 year", "0", regex=True)
    df[column] = df[column].str.replace(r" years", "", regex=True)
    df[column] = df[column].str.replace(r" year", "", regex=True)
    df[column] = pd.to_numeric(df[column])
    df[column].fillna(value=0, inplace=True)


def term_numeric(df, column):
    df[column] = pd.to_numeric(df[column].str.replace(" months", ""))


def date_columns(df, column):
    today_date = pd.to_datetime("2020-08-01")
    df[column] = pd.to_datetime(df[column], format="%b-%y")
    df["mths_since_" + column] = round(pd.to_numeric((today_date - df[column]) / np.timedelta64(1, "M")))
    df["mths_since_" + column] = df["mths_since_" + column].apply(
        lambda x: df["mths_since_" + column].max() if x < 0 else x
    )
    df.drop(columns=[column], inplace=True)


if "emp_length" in data.columns:
    emp_length_convert(data, "emp_length")

if "term" in data.columns:
    term_numeric(data, "term")

for col in ["issue_d", "last_pymnt_d", "last_credit_pull_d", "earliest_cr_line"]:
    if col in data.columns:
        date_columns(data, col)

categorical_features = data.select_dtypes(exclude="number")
numerical_features = data.select_dtypes(exclude="object")
preprocess_data = data.copy()

missing = preprocess_data.isnull().sum()
_ = missing[missing > 0]


# -----------------------------
# IV / WoE function (as in notebook)
# -----------------------------
def iv_woe(dataframe, target, bins=10, show_woe=False):
    newDF, woeDF = pd.DataFrame(), pd.DataFrame()
    cols = dataframe.columns

    for ivars in cols[~cols.isin([target])]:
        if (dataframe[ivars].dtype.kind in "bifc") and (len(np.unique(dataframe[ivars])) > 10):
            binned_x = pd.qcut(dataframe[ivars], bins, duplicates="drop")
            d0 = pd.DataFrame({"x": binned_x, "y": dataframe[target]})
        else:
            d0 = pd.DataFrame({"x": dataframe[ivars], "y": dataframe[target]})

        d = d0.groupby("x", as_index=False).agg({"y": ["count", "sum"]})
        d.columns = ["Cutoff", "N", "Events"]
        d["% of Events"] = np.maximum(d["Events"], 0.5) / d["Events"].sum()
        d["Non-Events"] = d["N"] - d["Events"]
        d["% of Non-Events"] = np.maximum(d["Non-Events"], 0.5) / d["Non-Events"].sum()
        d["WoE"] = np.log(d["% of Events"] / d["% of Non-Events"])
        d["IV"] = d["WoE"] * (d["% of Events"] - d["% of Non-Events"])
        d.insert(loc=0, column="Variable", value=ivars)

        print("Information value of " + ivars + " is " + str(round(d["IV"].sum(), 6)))
        temp = pd.DataFrame({"Variable": [ivars], "IV": [d["IV"].sum()]}, columns=["Variable", "IV"])
        newDF = pd.concat([newDF, temp], axis=0)
        woeDF = pd.concat([woeDF, d], axis=0)

        if show_woe:
            print(d)

    return newDF, woeDF


iv, woe = iv_woe(preprocess_data, target="bad_loan", bins=20)


# -----------------------------
# Drop low-IV and suspicious-IV columns (as in notebook)
# -----------------------------
cols_drop_iv = [
    "pymnt_plan", "last_pymnt_amnt", "revol_util", "delinq_2yrs", "mths_since_last_delinq",
    "open_acc", "pub_rec", "collections_12_mths_ex_med", "acc_now_delinq",
    "tot_coll_amt", "mths_since_last_pymnt_d", "emp_length", "application_type"
]
cols_drop_iv = [c for c in cols_drop_iv if c in preprocess_data.columns]
preprocess_data.drop(columns=cols_drop_iv, axis=1, inplace=True)


# -----------------------------
# Dummy variables creation (as in notebook)
# -----------------------------
data_dummies1 = [
    pd.get_dummies(preprocess_data["grade"], prefix="grade", prefix_sep=":") if "grade" in preprocess_data.columns else pd.DataFrame(),
    pd.get_dummies(preprocess_data["home_ownership"], prefix="home_ownership", prefix_sep=":") if "home_ownership" in preprocess_data.columns else pd.DataFrame(),
    pd.get_dummies(preprocess_data["verification_status"], prefix="verification_status", prefix_sep=":") if "verification_status" in preprocess_data.columns else pd.DataFrame(),
    pd.get_dummies(preprocess_data["purpose"], prefix="purpose", prefix_sep=":") if "purpose" in preprocess_data.columns else pd.DataFrame(),
    pd.get_dummies(preprocess_data["addr_state"], prefix="addr_state", prefix_sep=":") if "addr_state" in preprocess_data.columns else pd.DataFrame(),
    pd.get_dummies(preprocess_data["initial_list_status"], prefix="initial_list_status", prefix_sep=":") if "initial_list_status" in preprocess_data.columns else pd.DataFrame(),
]


def woe_categorical(df, cat_feature, good_bad_df):
    df = pd.concat([df[cat_feature], good_bad_df], axis=1)
    df = pd.concat(
        [
            df.groupby(df.columns.values[0], as_index=False)[df.columns.values[1]].count(),
            df.groupby(df.columns.values[0], as_index=False)[df.columns.values[1]].mean(),
        ],
        axis=1,
    )
    df = df.iloc[:, [0, 1, 3]]
    df.columns = [df.columns.values[0], "n_obs", "prop_good"]
    df["prop_n_obs"] = df["n_obs"] / df["n_obs"].sum()
    df["n_good"] = df["prop_good"] * df["n_obs"]
    df["n_bad"] = (1 - df["prop_good"]) * df["n_obs"]
    df["prop_n_good"] = df["n_good"] / df["n_good"].sum()
    df["prop_n_bad"] = df["n_bad"] / df["n_bad"].sum()
    df["WoE"] = np.log(df["prop_n_good"] / df["prop_n_bad"])
    df = df.sort_values(["WoE"]).reset_index(drop=True)
    df["diff_prop_good"] = df["prop_good"].diff().abs()
    df["diff_WoE"] = df["WoE"].diff().abs()
    df["IV"] = (df["prop_n_good"] - df["prop_n_bad"]) * df["WoE"]
    df["IV"] = df["IV"].sum()
    return df


def plot_by_woe(df_WoE, rotation_of_x_axis_labels=0):
    x = np.array(df_WoE.iloc[:, 0].apply(str))
    y = df_WoE["WoE"]
    plt.figure(figsize=(18, 12))
    plt.plot(
        x, y,
        marker="o",
        color="hotpink",
        linestyle="dashed",
        linewidth=3,
        markersize=18,
        markeredgecolor="cyan",
        markerfacecolor="black"
    )
    plt.xlabel(df_WoE.columns[0])
    plt.ylabel("Weight of Evidence")
    plt.title(str("Weight of Evidence by " + df_WoE.columns[0]))
    plt.xticks(rotation=rotation_of_x_axis_labels)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.show()


# -----------------------------
# Separate X/y again (as in notebook)
# -----------------------------
X = preprocess_data.drop(columns="bad_loan", axis=1)
y = preprocess_data["bad_loan"]


# -----------------------------
# WoE for continuous variables
# -----------------------------
def woe_continous(df, cat_feature, good_bad_df):
    df = pd.concat([df[cat_feature], good_bad_df], axis=1)
    df = pd.concat(
        [
            df.groupby(df.columns.values[0], as_index=False)[df.columns.values[1]].count(),
            df.groupby(df.columns.values[0], as_index=False)[df.columns.values[1]].mean(),
        ],
        axis=1,
    )
    df = df.iloc[:, [0, 1, 3]]
    df.columns = [df.columns.values[0], "n_obs", "prop_good"]
    df["prop_n_obs"] = df["n_obs"] / df["n_obs"].sum()
    df["n_good"] = df["prop_good"] * df["n_obs"]
    df["n_bad"] = (1 - df["prop_good"]) * df["n_obs"]
    df["prop_n_good"] = df["n_good"] / df["n_good"].sum()
    df["prop_n_bad"] = df["n_bad"] / df["n_bad"].sum()
    df["WoE"] = np.log(df["prop_n_good"] / df["prop_n_bad"])
    df["diff_prop_good"] = df["prop_good"].diff().abs()
    df["diff_WoE"] = df["WoE"].diff().abs()
    df["IV"] = (df["prop_n_good"] - df["prop_n_bad"]) * df["WoE"]
    df["IV"] = df["IV"].sum()
    return df


# Example from notebook (months since issue date factor)
if "mths_since_issue_d" in X.columns:
    X["mths_since_issue_d_factor"] = pd.cut(X["mths_since_issue_d"], 10)
    mths_since_iss_df = woe_continous(X, "mths_since_issue_d_factor", y)
    plot_by_woe(mths_since_iss_df)


# -----------------------------
# NOTE:
# The pasted text references "new_df" that contains the manually created WoE-binned dummy variables.
# That construction code is NOT included in your paste.
#
# We can still proceed by building a basic model on available numeric + one-hot categorical columns:
# -----------------------------
cat_cols = preprocess_data.select_dtypes(include=["object"]).columns.tolist()
num_cols = preprocess_data.select_dtypes(exclude=["object"]).columns.tolist()

# remove target from num_cols if present
if "bad_loan" in num_cols:
    num_cols.remove("bad_loan")

X_num = preprocess_data[num_cols].copy()
X_cat = pd.get_dummies(preprocess_data[cat_cols], drop_first=True) if len(cat_cols) > 0 else pd.DataFrame(index=preprocess_data.index)

new_df = pd.concat([X_num, X_cat, preprocess_data["bad_loan"]], axis=1)

# Dummy trap prevention (from notebook) using their ref_categories list (only drop those that exist)
ref_categories = [
    "home_ownership:OTHER_NONE_RENT_ANY", "total_rec_int:<1000", "total_pymnt:<5000", "total_rev_hi_lim:<10000",
    "grade:G", "verification_status:VERIFIED", "purpose:SMALL_BUSINESS_EDUCATIONAL_RENEWABLE_ENERGY_MOVING",
    "addr_state:NE_IA_NV_HI_FL_AL", "initial_list_status:F", "term:60", "mths_since_issue_d_:>90.4-96",
    "int_rate:21.932-26.06", "dti:27-40", "annual_inc:<32000", "inq_last_6mths:4-7", "tot_cur_bal:<40000",
    "mths_since_last_credit_pull_d:>76", "out_prncp:>12000"
]
ref_categories = [c for c in ref_categories if c in new_df.columns]
if len(ref_categories) > 0:
    new_df.drop(columns=ref_categories, inplace=True)


# -----------------------------
# Class balance visualization (optional)
# -----------------------------
X_final = new_df.drop(columns="bad_loan", axis=1)
y_final = new_df["bad_loan"]

if HAS_YELLOWBRICK:
    visualizer = ClassBalance()
    visualizer.fit(y_final)
    visualizer.show()


# -----------------------------
# Train/test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.2, random_state=42)


# -----------------------------
# Oversample minority class
# -----------------------------
os = RandomOverSampler(random_state=42)
X_train_o, y_train_o = os.fit_resample(X_train, y_train)


# -----------------------------
# Logistic regression PD model
# -----------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train_o, y_train_o)

y_preds = model.predict(X_test)
print(classification_report(y_test, y_preds))


# -----------------------------
# Discriminatory power metrics
# -----------------------------
# ROC + AUC
y_scores = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_scores)
print("AUC:", auc)

# Gini
gini = 2 * auc - 1
print("Gini Index:", gini)

# Precision-Recall
ap = average_precision_score(y_test, y_scores)
print("Precision-Recall Score (Average Precision):", ap)

# KS Statistic
fpr, tpr, thresholds = roc_curve(y_test, y_scores)
ks = np.max(tpr - fpr)
print("KS score is", ks)

# Plot ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, linewidth=2)
plt.plot([0, 1], [0, 1], linestyle="--", linewidth=2)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.tight_layout()
plt.show()

# Plot Precision-Recall
precision, recall, _ = precision_recall_curve(y_test, y_scores)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, linewidth=2)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.tight_layout()
plt.show()


# -----------------------------
# Save model
# -----------------------------
import pickle
filename = "credit_risk_PD_model.sav"
pickle.dump(model, open(filename, "wb"))
print("Saved model to:", filename)