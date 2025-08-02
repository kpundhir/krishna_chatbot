import os
import pandas as pd

# Use relative path to load the correct cleaned CSV
DATA_PATH = os.path.join("data", "bhagavad_gita_cleaned.csv")


def main():
    print("🔹 Loading cleaned Bhagavad Gita dataset...")

    # Load the CSV file
    df = pd.read_csv(DATA_PATH)

    print(f"✅ Loaded {len(df)} verses.")
    print("\n🔍 Sample:")
    print(df[["chapter", "verse", "citation_text"]].head())


if __name__ == "__main__":
    main()
