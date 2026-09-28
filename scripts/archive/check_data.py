"""Скрипт для проверки данных"""
import pandas as pd
from pathlib import Path

# Укажите путь к вашему файлу
FILE_PATH = "../data/raw/your_data.csv"  # ← ИЗМЕНИТЕ

print("Checking file...")

# Попытка загрузки с разными параметрами
encodings = ['utf-8', 'utf-16', 'utf-16-le', 'cp1251', 'latin1']
separators = ['\t', ',', ';']

for enc in encodings:
    for sep in separators:
        try:
            df = pd.read_csv(FILE_PATH, encoding=enc, sep=sep, nrows=5)
            if len(df.columns) > 1:
                print(f"\n✓ SUCCESS with encoding='{enc}', separator='{repr(sep)}'")
                print(f"\nColumns ({len(df.columns)}):")
                for i, col in enumerate(df.columns):
                    print(f"  {i+1}. {col}")
                print(f"\nFirst row:")
                print(df.iloc[0])
                exit(0)
        except Exception as e:
            pass

# Попробуем как Excel
try:
    df = pd.read_excel(FILE_PATH, nrows=5)
    print(f"\n✓ SUCCESS as Excel file")
    print(f"\nColumns ({len(df.columns)}):")
    for i, col in enumerate(df.columns):
        print(f"  {i+1}. {col}")
    exit(0)
except:
    pass

print("\n✗ Could not load file!")