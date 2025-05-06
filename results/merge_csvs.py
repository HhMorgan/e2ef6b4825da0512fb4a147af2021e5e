import pandas as pd

# Load CSVs
seq = pd.read_csv("results_seq.csv")
scf = pd.read_csv("results_scf.csv")
mcf = pd.read_csv("results_mcf.csv")

# Rename runtime columns
seq.rename(columns={"runtime": "SEQ"}, inplace=True)
scf.rename(columns={"runtime": "SCF"}, inplace=True)
mcf.rename(columns={"runtime": "MCF"}, inplace=True)

# Merge on instance and k
merged = seq.merge(scf, on=["instance", "k"]).merge(mcf, on=["instance", "k"])

# Save for LaTeX
merged.to_csv("merged_results.csv", index=False)