"""Export one baseline table to publication-friendly formats."""

import pandas as pd

from openstata import OpenStata

patients = pd.DataFrame(
    {
        "arm": ["Control", "Control", "Control", "Treatment", "Treatment", "Treatment"],
        "age": [64, 58, 71, 61, 55, 68],
        "crp": [4.2, 13.1, 7.4, 3.8, 8.5, 5.1],
        "sex": ["Female", "Male", "Female", "Female", "Female", "Male"],
    }
)

stata = OpenStata(patients)
common = {
    "by": "arm",
    "categorical": ["sex"],
    "nonnormal": ["crp"],
    "labels": {"age": "Age, years", "crp": "CRP, mg/L", "sex": "Sex"},
    "pvalues": True,
    "standardized_differences": True,
}

stata.export_table1(
    "baseline.html",
    ["age", "crp", "sex"],
    title="Table 1. Participant characteristics",
    subtitle="Synthetic demonstration cohort",
    style="clinical",
    **common,
)
stata.export_table1(
    "baseline.xlsx",
    ["age", "crp", "sex"],
    title="Table 1. Participant characteristics",
    subtitle="Synthetic demonstration cohort",
    style="clinical",
    **common,
)
