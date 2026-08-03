"""Small end-to-end OpenStata example."""

import pandas as pd

from openstata import OpenStata, table1

patients = pd.DataFrame(
    {
        "arm": ["Control", "Control", "Treatment", "Treatment"],
        "age": [64, 58, 61, 55],
        "crp": [4.2, 13.1, 3.8, 8.5],
        "female": [1, 0, 1, 1],
    }
)

stata = OpenStata(patients)
print(stata.run("summarize age crp, detail"))
print(stata.run("tabulate arm female, row"))
print(
    table1(
        patients,
        ["age", "crp", "female"],
        by="arm",
        categorical=["female"],
        nonnormal=["crp"],
        pvalues=True,
        standardized_differences=True,
    )
)
