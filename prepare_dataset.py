import pandas as pd

good_files = [
    "data/good1.csv",
    "data/good2.csv",
    "data/good3.csv"
]

bad_files = [
    "data/bad1.csv",
    "data/bad2.csv",
    "data/bad3.csv"
]


def load_and_label(files, label):
    dfs = []
    for file in files:
        df = pd.read_csv(file)
        df["label"] = label
        dfs.append(df)
    return dfs

# load good posture CSVs
good_dfs = load_and_label(good_files, "good")

# load bad posture CSVs
bad_dfs = load_and_label(bad_files, "bad")

# combine all dfs
full_df = pd.concat(good_dfs + bad_dfs, ignore_index=True)

# shuffle
full_df = full_df.sample(frac=1, random_state=42).reset_index(drop=True)

# save final dataset
#full_df.to_csv("dataset.csv", index=False)

print("Total samples:", len(full_df))
print("Good:", sum(full_df.label == "good"))
print("Bad:", sum(full_df.label == "bad"))