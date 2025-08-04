import os
import pandas as pd

DATA_PATH = os.path.join("data", "bhagavad_gita_cleaned.csv")

def main():
    df = pd.read_csv(DATA_PATH)
    print(f"{len(df)} verses loaded.")
    print(df[["chapter", "verse", "citation_text"]].head())

if __name__ == "__main__":
    main()
