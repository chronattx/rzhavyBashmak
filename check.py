import pandas as pd
import numpy as np
import glob


def load_deficit(file_path):
    df = pd.read_csv(file_path)
    return df[df['deficit'] > 0]  # Только положительный дефицит


def load_new_lamps(folder_path):
    lamp_data = {}
    for file in glob.glob(f"{folder_path}/*.txt"):
        name = file.split("/")[-1].split("\\")[-1].split(".")[0]
        df = pd.read_csv(file, sep=" ", skiprows=1, header=None)
        df.columns = ['wavelength', 'intensity']
        lamp_data[name] = df
    return lamp_data


def check_lamp_coverage(deficit_df, lamp_df, threshold=0.01):
    """Проверка, сколько точек дефицита покрываются лампой"""
    merged = pd.merge(deficit_df, lamp_df, on='wavelength', how='inner', suffixes=('_deficit', '_lamp'))
    covered = merged[merged['intensity'] >= threshold]
    return len(covered), covered


def main():
    deficit_df = load_deficit("deficit_report.csv")
    lamp_dict = load_new_lamps("newlamps")

    print(f"🔎 Всего нехваток: {len(deficit_df)} точек спектра\n")

    coverage_results = []

    for name, df in lamp_dict.items():
        count, covered = check_lamp_coverage(deficit_df, df)
        coverage_results.append((name, count))

    # Сортировка по количеству покрытых точек (по убыванию)
    coverage_results.sort(key=lambda x: x[1], reverse=True)

    print("✅ Лампы, отсортированные по количеству покрытых точек:")
    for name, count in coverage_results:
        status = "✅" if count > 0 else "❌"
        print(f" {status} {name}: покрывает {count} точек")


if __name__ == "__main__":
    main()
