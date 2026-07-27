import os
import pandas as pd
import pytse_client as tse

from symbols import SYMBOLS

os.makedirs("finetune_csv/data", exist_ok=True)

for eng_name, fa_symbol in SYMBOLS.items():
    print(f"Downloading {eng_name}...")
    try:
        tse.download(
            symbols=fa_symbol,
            write_to_csv=True,
            base_path="finetune_csv/data"
        )
        # rename فارسی به انگلیسی
        fa_path = f"finetune_csv/data/{fa_symbol}.csv"
        eng_path = f"finetune_csv/data/{eng_name}.csv"
        if os.path.exists(fa_path):
            os.rename(fa_path, eng_path)
        print(f"[OK] {eng_name} saved")
    except Exception as e:
        print(f"[ERROR] {eng_name}: {e}")


# Rewrite the CSV files to ensure they are in the correct format
print("Rewriting CSV files...")
for eng_name in SYMBOLS.keys():
    eng_path = f"finetune_csv/data/{eng_name}.csv"
    if os.path.exists(eng_path):
        try:
            df = pd.read_csv(eng_path)
            
            # Map existing columns to the required format
            rename_map = {
                'date': 'timestamps',
                'value': 'amount'
            }
            df.rename(columns=rename_map, inplace=True)
            
            # Format the timestamps to match the sample in the README (YYYY/MM/DD HH:MM)
            df['timestamps'] = pd.to_datetime(df['timestamps']).dt.strftime('%Y/%m/%d 12:00')
            
            required_cols = ['timestamps', 'open', 'high', 'low', 'close', 'volume', 'amount']
            
            # Ensure only required columns are present in the final CSV
            available_cols = [col for col in required_cols if col in df.columns]
            df = df[available_cols]
            
            df.to_csv(eng_path, index=False)
            print(f"[OK] {eng_name} CSV rewritten correctly")
        except Exception as e:
            print(f"[ERROR] Failed to rewrite {eng_name}: {e}")

