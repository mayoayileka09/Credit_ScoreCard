from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from credit_scorecard.config.config import logger


# ----------------------------
# Helpers
# ----------------------------
def _safe_drop(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    cols = [c for c in cols if c in df.columns]
    if cols:
        df = df.drop(columns=cols)
    return df


def _emp_length_convert(s: pd.Series) -> pd.Series:
    # Matches the kernel logic
    s = s.astype("string")
    s = s.str.replace(r"\+ years", "", regex=True)
    s = s.str.replace(r"< 1 year", "0", regex=True)
    s = s.str.replace(r" years", "", regex=True)
    s = s.str.replace(r" year", "", regex=True)
    out = pd.to_numeric(s, errors="coerce").fillna(0).astype("int64")
    return out


def _term_numeric(s: pd.Series) -> pd.Series:
    s = s.astype("string")
    out = pd.to_numeric(s.str.replace(" months", "", regex=False), errors="coerce")
    return out.astype("Int64").fillna(0).astype("int64")


def _months_since(series_dt: pd.Series, today_date: pd.Timestamp) -> pd.Series:
    """
    Calendar-month difference between today_date and each datetime in series_dt.
    Equivalent to: (today.year - dt.year)*12 + (today.month - dt.month), with day-adjustment.
    """
    dt = pd.to_datetime(series_dt, errors="coerce")

    # Base month difference ignoring day-of-month
    months = (today_date.year - dt.dt.year) * 12 + (today_date.month - dt.dt.month)

    # Optional day-of-month adjustment:
    # If dt's day is after today's day, subtract 1 month because it's not a full month elapsed.
    # This better matches "months since" intuition.
    months = months - (dt.dt.day > today_date.day).astype("int64")

    return months.astype("float64")  # keep float for NaNs until filled


def _date_to_months_since(df: pd.DataFrame, col: str, today_date: pd.Timestamp) -> pd.DataFrame:
    if col not in df.columns:
        return df

    dt = pd.to_datetime(df[col], format="%b-%y", errors="coerce")
    m = _months_since(dt, today_date=today_date)

    new_col = f"mths_since_{col}"

    # Kernel behavior: replace negatives with max
    max_val = pd.to_numeric(m, errors="coerce").max()
    m = m.where(m >= 0, max_val)

    # Fill NaNs with max as well (keeps behavior consistent and avoids missing values)
    df[new_col] = m.fillna(max_val).round().astype("int64")

    df = df.drop(columns=[col])
    return df


# ----------------------------
# IV / WOE (for IV filtering only)
# ----------------------------
def _iv_woe_table(df: pd.DataFrame, target: str, bins: int = 20) -> pd.DataFrame:
    """
    Returns a DataFrame with columns: Variable, IV
    Uses the same approach as the kernel: qcut for continuous with >10 unique values.
    """
    iv_rows = []

    for col in [c for c in df.columns if c != target]:
        x = df[col]

        if x.dtype.kind in "bifc" and x.nunique(dropna=True) > 10:
            # qcut may drop duplicates
            binned = pd.qcut(x, q=bins, duplicates="drop")
            d0 = pd.DataFrame({"x": binned, "y": df[target]})
        else:
            d0 = pd.DataFrame({"x": x, "y": df[target]})

        d = d0.groupby("x", as_index=False).agg(y_count=("y", "count"), y_sum=("y", "sum"))
        d = d.rename(columns={"x": "Cutoff", "y_count": "N", "y_sum": "Events"})

        # Events = sum(y) where y is 0/1 (here 1=good, 0=bad in your kernel)
        # This matches the kernel’s math (it calls them Events/Non-Events regardless of semantics).
        d["% of Events"] = np.maximum(d["Events"], 0.5) / max(d["Events"].sum(), 1e-9)
        d["Non-Events"] = d["N"] - d["Events"]
        d["% of Non-Events"] = np.maximum(d["Non-Events"], 0.5) / max(d["Non-Events"].sum(), 1e-9)
        d["WoE"] = np.log(d["% of Events"] / d["% of Non-Events"])
        d["IV"] = d["WoE"] * (d["% of Events"] - d["% of Non-Events"])

        iv_rows.append({"Variable": col, "IV": float(d["IV"].sum())})

    return pd.DataFrame(iv_rows).sort_values("IV", ascending=False).reset_index(drop=True)


# ----------------------------
# Category grouping (as described in your notes)
# ----------------------------
def _group_home_ownership(s: pd.Series) -> pd.Series:
    s = s.astype("string").fillna("UNKNOWN")
    # OTHER, NONE, ANY -> group with RENT
    return s.replace({"OTHER": "OTHER_NONE_RENT_ANY", "NONE": "OTHER_NONE_RENT_ANY", "ANY": "OTHER_NONE_RENT_ANY", "RENT": "OTHER_NONE_RENT_ANY"})


def _group_purpose(s: pd.Series) -> pd.Series:
    s = s.astype("string").fillna("UNKNOWN")

    group_map = {
        # SMALL_BUSINESS_EDUCATIONAL_RENEWABLE_ENERGY_MOVING
        "small_business": "SMALL_BUSINESS_EDUCATIONAL_RENEWABLE_ENERGY_MOVING",
        "educational": "SMALL_BUSINESS_EDUCATIONAL_RENEWABLE_ENERGY_MOVING",
        "renewable_energy": "SMALL_BUSINESS_EDUCATIONAL_RENEWABLE_ENERGY_MOVING",
        "moving": "SMALL_BUSINESS_EDUCATIONAL_RENEWABLE_ENERGY_MOVING",
        # OTHER_HOUSE_MEDICAL
        "other": "OTHER_HOUSE_MEDICAL",
        "house": "OTHER_HOUSE_MEDICAL",
        "medical": "OTHER_HOUSE_MEDICAL",
        # WEDDING_VACATION
        "wedding": "WEDDING_VACATION",
        "vacation": "WEDDING_VACATION",
        # HOME_IMPROVEMENT_MAJOR_PURCHASE
        "home_improvement": "HOME_IMPROVEMENT_MAJOR_PURCHASE",
        "major_purchase": "HOME_IMPROVEMENT_MAJOR_PURCHASE",
        # CAR_CREDIT_CARD (your note lists car + credit_card; kernel note also mentions credit_card)
        "car": "CAR_CREDIT_CARD",
        "credit_card": "CAR_CREDIT_CARD",
        # Keep these as-is unless you want to force them into a group:
        # "debt_consolidation": "debt_consolidation",
    }
    return s.map(lambda v: group_map.get(v, v))


def _group_addr_state(s: pd.Series) -> pd.Series:
    s = s.astype("string").fillna("UNKNOWN")

    groups = {
        "NE_IA_NV_HI_FL_AL": {"NE", "IA", "NV", "HI", "FL", "AL"},
        "NY": {"NY"},
        "LA_NM_OK_NC_MO_MD_NJ_VA": {"LA", "NM", "OK", "NC", "MO", "MD", "NJ", "VA"},
        "CA": {"CA"},
        "AZ_MI_UT_TN_AR_PA": {"AZ", "MI", "UT", "TN", "AR", "PA"},
        "RI_OH_KY_DE_MN_SD_MA_IN": {"RI", "OH", "KY", "DE", "MN", "SD", "MA", "IN"},
        "GA_WA": {"GA", "WA"},
        "WI_OR": {"WI", "OR"},
        "TX": {"TX"},
        "IL_CT_MT": {"IL", "CT", "MT"},
        "CO_SC": {"CO", "SC"},
        "KS_VT_AK_MS": {"KS", "VT", "AK", "MS"},
        "NH_WV_WY_DC": {"NH", "WV", "WY", "DC"},
    }

    def mapper(v: str) -> str:
        for k, vs in groups.items():
            if v in vs:
                return k
        return v  # leave the rest as their own category (or UNKNOWN)

    return s.map(mapper)


def _bin_continuous_to_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates the “new_df”-style binned categorical features described in your notes.
    This is the main “reproducibility” piece—so the downstream dummy set matches.
    """
    out = df.copy()

    # term
    if "term" in out.columns:
        out["term"] = out["term"].astype(int)
        out["term"] = out["term"].map(lambda x: 36 if x == 36 else 60)

    # total_rec_int bins (from your new_df header)
    if "total_rec_int" in out.columns:
        x = out["total_rec_int"]
        out["total_rec_int_bin"] = pd.cut(
            x,
            bins=[-np.inf, 1000, 2000, 9000, np.inf],
            labels=["<1000", "1000-2000", "2000-9000", ">9000"],
        )

    # total_rev_hi_lim bins (your header had both <100000 and bucketed <100000; we’ll bucket + keep >100000 as >100000)
    if "total_rev_hi_lim" in out.columns:
        x = out["total_rev_hi_lim"]
        out["total_rev_hi_lim_bin"] = pd.cut(
            x,
            bins=[-np.inf, 10000, 20000, 40000, 60000, 80000, 100000, np.inf],
            labels=[
                "<10000",
                "10000-20000",
                "20000-40000",
                "40000-60000",
                "60000-80000",
                "80000-100000",
                ">100000",
            ],
        )

    # total_pymnt bins
    if "total_pymnt" in out.columns:
        x = out["total_pymnt"]
        out["total_pymnt_bin"] = pd.cut(
            x,
            bins=[-np.inf, 5000, 11000, 16000, 22000, np.inf],
            labels=["<5000", "5000-11000", "11000-16000", "16000-22000", ">22000"],
        )

    # int_rate bins (from your header / notes)
    if "int_rate" in out.columns:
        x = out["int_rate"]
        out["int_rate_bin"] = pd.cut(
            x,
            bins=[-np.inf, 7.484, 9.548, 11.612, 13.676, 15.74, 17.804, 19.868, 21.932, np.inf],
            labels=[
                "<7.484",
                "7.484-9.548",
                "9.548-11.612",
                "11.612-13.676",
                "13.676-15.74",
                "15.74-17.804",
                "17.804-19.868",
                "19.868-21.932",
                "21.932-26.06",
            ],
        )

    # dti bins (merge top 3 into 27-40)
    if "dti" in out.columns:
        x = out["dti"]
        out["dti_bin"] = pd.cut(
            x,
            bins=[-np.inf, 4, 8, 12, 16, 20, 23, 27, np.inf],
            labels=["<4", "4-8", "8-12", "12-16", "16-20", "20-23", "23-27", "27-40"],
        )

    # annual_inc bins (<=32000, 32000-50000, ..., >150000)
    if "annual_inc" in out.columns:
        x = out["annual_inc"]
        out["annual_inc_bin"] = pd.cut(
            x,
            bins=[-np.inf, 32000, 50000, 60000, 75000, 90000, 120000, 135000, 150000, np.inf],
            labels=[
                "<32000",
                "32000-50000",
                "50000-60000",
                "60000-75000",
                "75000-90000",
                "90000-120000",
                "120000-135000",
                "135000-150000",
                ">150000",
            ],
        )

    # inq_last_6mths bins (<1, 1-2, 2-4, 4-7)
    if "inq_last_6mths" in out.columns:
        x = out["inq_last_6mths"]
        out["inq_last_6mths_bin"] = pd.cut(
            x,
            bins=[-np.inf, 1, 2, 4, np.inf],
            labels=["<1", "1-2", "2-4", "4-7"],
            right=False,
        )

    # tot_cur_bal bins (<40000, ..., 320000-400000, >400000)
    if "tot_cur_bal" in out.columns:
        x = out["tot_cur_bal"]
        out["tot_cur_bal_bin"] = pd.cut(
            x,
            bins=[-np.inf, 40000, 80000, 120000, 160000, 200000, 240000, 320000, 400000, np.inf],
            labels=[
                "<40000",
                "40000-80000",
                "80000-120000",
                "120000-160000",
                "160000-200000",
                "200000-240000",
                "240000-320000",
                "320000-400000",
                ">400000",
            ],
        )

    # mths_since_last_credit_pull_d bins (<65, 65-76, >76)
    if "mths_since_last_credit_pull_d" in out.columns:
        x = out["mths_since_last_credit_pull_d"]
        out["mths_since_last_credit_pull_d_bin"] = pd.cut(
            x,
            bins=[-np.inf, 65, 76, np.inf],
            labels=["<65", "65-76", ">76"],
        )

    # mths_since_issue_d bins (matching your final categories)
    if "mths_since_issue_d" in out.columns:
        x = out["mths_since_issue_d"]
        out["mths_since_issue_d_bin"] = pd.cut(
            x,
            bins=[-np.inf, 70.8, 73.6, 76.4, 79.2, 82, 84, 90.4, 96, np.inf],
            labels=[
                "<70.8",
                ">70.8-73.6",
                "73.6-76.4",
                ">76.4-79.2",
                ">79.2-82",
                ">82-84",
                ">84-90.4",
                ">90.4-96",
                ">96",
            ],
        )

    # out_prncp bins (from your header)
    if "out_prncp" in out.columns:
        x = out["out_prncp"]
        out["out_prncp_bin"] = pd.cut(
            x,
            bins=[-np.inf, 3000, 6000, 10000, 12000, np.inf],
            labels=["<3000", "3000-6000", "6000-10000", "10000-12000", ">12000"],
        )

    return out


# ----------------------------
# Main preprocessing pipeline
# ----------------------------
def preprocess_data(data_path: str, clean_path: str | None = None) -> str:
    """
    Reproduces the LendingClub PD-model kernel-style preprocessing and saves a model-ready CSV.

    Output includes:
      - bad_loan (0=bad, 1=good) as target
      - one-hot encoded grouped categories
      - binned continuous features (one-hot)
      - reference dummies removed (dummy trap)
    """
    df = pd.read_csv(data_path)
    logger.info(f"Loaded data shape: {df.shape}")

    # -------------------------------------------------------
    # 1) Target creation (kernel logic: 0 bad, 1 good)
    # -------------------------------------------------------
    if "loan_status" not in df.columns:
        raise ValueError("Expected column 'loan_status' not found in input CSV.")

    bad_statuses = {
        "Charged Off",
        "Default",
        "Late (31-120 days)",
        "Does not meet the credit policy. Status:Charged Off",
    }
    df["bad_loan"] = np.where(df["loan_status"].isin(list(bad_statuses)), 0, 1).astype("int64")

    # Drop original loan_status + Unnamed index col if present
    df = _safe_drop(df, ["loan_status", "Unnamed: 0"])

    # -------------------------------------------------------
    # 2) Drop high-missing + IDs + leakage/future columns
    # (matches your columns_to_drop list; safe if absent)
    # -------------------------------------------------------
    columns_to_drop = [
        "id",
        "member_id",
        "sub_grade",
        "emp_title",
        "url",
        "desc",
        "title",
        "zip_code",
        "next_pymnt_d",
        "recoveries",
        "collection_recovery_fee",
        "total_rec_prncp",
        "total_rec_late_fee",
        "mths_since_last_record",
        "mths_since_last_major_derog",
        "annual_inc_joint",
        "dti_joint",
        "verification_status_joint",
        "open_acc_6m",
        "open_il_6m",
        "open_il_12m",
        "open_il_24m",
        "mths_since_rcnt_il",
        "total_bal_il",
        "il_util",
        "open_rv_12m",
        "open_rv_24m",
        "max_bal_bc",
        "all_util",
        "inq_fi",
        "total_cu_tl",
        "inq_last_12m",
        "policy_code",
    ]
    df = _safe_drop(df, columns_to_drop)

    # -------------------------------------------------------
    # 3) Drop remaining missing rows (kernel does dropna here)
    # -------------------------------------------------------
    before = df.shape
    df = df.dropna()
    logger.info(f"After initial dropna (post-column-drop): {before} -> {df.shape}")

    # -------------------------------------------------------
    # 4) Correlation-based / multicollinar drop (your list)
    # -------------------------------------------------------
    multicollinear_drop = [
        "loan_amnt",
        "revol_bal",
        "funded_amnt",
        "funded_amnt_inv",
        "installment",
        "total_pymnt_inv",
        "out_prncp_inv",
        "total_acc",
    ]
    df = _safe_drop(df, multicollinear_drop)

    # -------------------------------------------------------
    # 5) Type transforms (emp_length, term, dates -> months since)
    # -------------------------------------------------------
    if "emp_length" in df.columns:
        df["emp_length"] = _emp_length_convert(df["emp_length"])
    if "term" in df.columns:
        df["term"] = _term_numeric(df["term"])

    today_date = pd.to_datetime("2020-08-01")
    for date_col in ["issue_d", "last_pymnt_d", "last_credit_pull_d", "earliest_cr_line"]:
        df = _date_to_months_since(df, date_col, today_date=today_date)

    # -------------------------------------------------------
    # 6) IV calculation (for filtering like kernel)
    # -------------------------------------------------------
    # Kernel rule-of-thumb: IV < 0.02 useless; IV > 0.5 suspicious.
    # You also listed a specific set to drop; we follow your list for reproducibility.
    iv_table = _iv_woe_table(df, target="bad_loan", bins=20)
    logger.info("Top IV features:\n" + str(iv_table.head(10)))

    # Your explicit drop list after IV step (keep consistent with your writeup)
    iv_based_drop = [
        "pymnt_plan",
        "last_pymnt_amnt",
        "revol_util",
        "delinq_2yrs",
        "mths_since_last_delinq",
        "open_acc",
        "pub_rec",
        "collections_12_mths_ex_med",
        "acc_now_delinq",
        "tot_coll_amt",
        "mths_since_last_pymnt_d",
        "emp_length",
        "application_type",
    ]
    df = _safe_drop(df, iv_based_drop)

    # -------------------------------------------------------
    # 7) Category grouping (home_ownership, purpose, addr_state)
    # -------------------------------------------------------
    if "home_ownership" in df.columns:
        df["home_ownership_grp"] = _group_home_ownership(df["home_ownership"])
    if "purpose" in df.columns:
        df["purpose_grp"] = _group_purpose(df["purpose"])
    if "addr_state" in df.columns:
        df["addr_state_grp"] = _group_addr_state(df["addr_state"])

    # Keep original grade / verification / initial_list_status as-is (kernel did)
    # We’ll drop raw columns after creating grouped variants where applicable
    df = _safe_drop(df, ["home_ownership", "purpose", "addr_state"])

    # -------------------------------------------------------
    # 8) Continuous binning into categories (then one-hot)
    # -------------------------------------------------------
    df = _bin_continuous_to_categories(df)

    # -------------------------------------------------------
    # 9) One-hot encode: grade, verification_status, initial_list_status,
    #                   and grouped categories + binned continuous categories
    # -------------------------------------------------------
    cat_cols = []

    for c in [
        "grade",
        "verification_status",
        "initial_list_status",
        "home_ownership_grp",
        "purpose_grp",
        "addr_state_grp",
        "term",
        "total_rec_int_bin",
        "total_rev_hi_lim_bin",
        "total_pymnt_bin",
        "int_rate_bin",
        "dti_bin",
        "annual_inc_bin",
        "inq_last_6mths_bin",
        "tot_cur_bal_bin",
        "mths_since_last_credit_pull_d_bin",
        "mths_since_issue_d_bin",
        "out_prncp_bin",
    ]:
        if c in df.columns:
            cat_cols.append(c)

    dummies = []
    for c in cat_cols:
        prefix = c
        d = pd.get_dummies(df[c], prefix=prefix, prefix_sep=":", dummy_na=False)
        dummies.append(d)

    X_cat = pd.concat(dummies, axis=1) if dummies else pd.DataFrame(index=df.index)

    # Base numeric columns = everything not target and not the raw categorical columns we encoded
    drop_after_encode = set(cat_cols)
    keep_numeric = df.drop(columns=list(drop_after_encode), errors="ignore")

    # Assemble final
    final_df = pd.concat([keep_numeric.drop(columns=["bad_loan"], errors="ignore"), X_cat, df["bad_loan"]], axis=1)

    # -------------------------------------------------------
    # 10) Drop reference categories (dummy trap) using your list
    # Names must match the prefixes we used above.
    # Some may not exist depending on data; safe drop.
    # -------------------------------------------------------
    ref_categories = [
        "home_ownership_grp:OTHER_NONE_RENT_ANY",
        "total_rec_int_bin:<1000",
        "total_pymnt_bin:<5000",
        "total_rev_hi_lim_bin:<10000",
        "grade:G",
        "verification_status:VERIFIED",
        "purpose_grp:SMALL_BUSINESS_EDUCATIONAL_RENEWABLE_ENERGY_MOVING",
        "addr_state_grp:NE_IA_NV_HI_FL_AL",
        "initial_list_status:F",
        "term:60",
        "mths_since_issue_d_bin:>90.4-96",
        "int_rate_bin:21.932-26.06",
        "dti_bin:27-40",
        "annual_inc_bin:<32000",
        "inq_last_6mths_bin:4-7",
        "tot_cur_bal_bin:<40000",
        "mths_since_last_credit_pull_d_bin:>76",
        "out_prncp_bin:>12000",
    ]
    final_df = _safe_drop(final_df, ref_categories)

    # -------------------------------------------------------
    # 11) Final sanity checks
    # -------------------------------------------------------
    if "bad_loan" not in final_df.columns:
        raise RuntimeError("Preprocessing failed: target 'bad_loan' missing from final dataset.")

    # Drop any remaining NA just in case transformations introduced them
    before = final_df.shape
    final_df = final_df.dropna()
    logger.info(f"Final dropna safety pass: {before} -> {final_df.shape}")

    # -------------------------------------------------------
    # 12) Save
    # -------------------------------------------------------
    if clean_path is None:
        clean_path = data_path.replace(".csv", "_preprocessed.csv")

    final_df.to_csv(clean_path, index=False)
    logger.info(f"Saved fully preprocessed dataset to {clean_path}")
    logger.info(f"Final dataset shape: {final_df.shape}")

    return clean_path