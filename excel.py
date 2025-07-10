import pandas as pd

def extract_and_transform_excel(file_path, output_path):
    # Load workbook
    xls = pd.ExcelFile(file_path)
    all_data = []

    for sheet_name in xls.sheet_names:
        df = xls.parse(sheet_name, header=None)
        
        try:
            # Select rows 1-12 and row 74 (0-indexed), from 2nd column onward
            df_selected = pd.concat([
                df.iloc[0:12, 1:],   # rows 1 to 12
                df.iloc[[73], 1:]    # row 74
            ])

            # Use the first column of those rows as row labels
            new_columns = df.iloc[[i for i in range(0, 12)] + [73], 0].tolist()
            df_selected.index = new_columns

            # Transpose and reset index
            df_transposed = df_selected.T.reset_index(drop=True)

            # Add a column to track the source sheet name
            df_transposed["Sheet Name"] = sheet_name

            all_data.append(df_transposed)

        except Exception as e:
            print(f"⚠️ Skipping sheet '{sheet_name}' due to error: {e}")
            continue

    # Combine all transformed data
    combined_df = pd.concat(all_data, ignore_index=True)

    # Save to Excel with one sheet
    combined_df.to_excel(output_path, index=False, sheet_name='CombinedData')
    print(f"✅ Data written to '{output_path}' with sheet names included.")

# Example usage
extract_and_transform_excel("input_workbook.xlsx", "combined_output.xlsx")
