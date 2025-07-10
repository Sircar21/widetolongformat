import pandas as pd

def extract_and_transform_excel(file_path, output_path):
    xls = pd.ExcelFile(file_path)
    sheet_names = xls.sheet_names[2:]  # skip first two template sheets

    all_data = []
    reference_columns = None  # will be set after processing the first valid sheet

    for i, sheet_name in enumerate(sheet_names):
        df = xls.parse(sheet_name, header=None)

        try:
            # Extract rows 1–12 and row 74, second column onwards
            df_selected = pd.concat([
                df.iloc[0:12, 1:],  # rows 1 to 12
                df.iloc[[73], 1:]   # row 74
            ])

            # Use column A (col 0) as labels for rows (to become columns after transpose)
            labels = df.iloc[[i for i in range(0, 12)] + [73], 0].tolist()
            
            # Fix: ensure labels are unique to prevent reindexing error
            unique_labels = pd.io.parsers.ParserBase({'names': labels})._maybe_dedup_names(labels)
            df_selected.index = unique_labels

            # Transpose
            df_transposed = df_selected.T.reset_index(drop=True)

            # Set standard columns from the first non-template sheet
            if reference_columns is None:
                reference_columns = unique_labels  # capture once
            df_transposed.columns = reference_columns  # ensure consistent column names

            # Add sheet name
            df_transposed["Sheet Name"] = sheet_name

            all_data.append(df_transposed)

        except Exception as e:
            print(f"⚠️ Skipping sheet '{sheet_name}' due to error: {e}")
            continue

    # Combine all transformed data
    combined_df = pd.concat(all_data, ignore_index=True)

    # Export to Excel
    combined_df.to_excel(output_path, index=False, sheet_name='CombinedData')
    print(f"✅ Data written to '{output_path}' with all sheets processed (except first 2).")
