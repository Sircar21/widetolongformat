import pandas as pd
import io

def drop_rows_based_on_conditions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops rows from the DataFrame where:
    1. The 'Company Type' column is not 'Y'.
    2. The 'Cac/Raw', 'Remarks', and 'Marking' columns are all empty (NaN, None, or empty string).

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    pd.DataFrame: A new DataFrame with the specified rows dropped.
    """
    # Ensure empty strings and spaces are treated as NaN for consistent checking
    df_cleaned = df.replace({'': pd.NA, ' ': pd.NA})

    # Define the conditions for dropping rows
    condition_company_type = (df_cleaned['Company Type'] != 'Y')
    condition_cac_raw_empty = df_cleaned['Cac/Raw'].isna()
    condition_remarks_empty = df_cleaned['Remarks'].isna()
    condition_marking_empty = df_cleaned['Marking'].isna()

    # Combine conditions: drop if Company Type is not 'Y' AND all three specified columns are empty
    rows_to_drop = condition_company_type & \
                   condition_cac_raw_empty & \
                   condition_remarks_empty & \
                   condition_marking_empty

    # Filter the DataFrame to keep rows that DO NOT meet the dropping conditions
    df_filtered = df_cleaned[~rows_to_drop].copy()

    # Restore None for cells that were originally empty but replaced with pd.NA for filtering
    df_filtered = df_filtered.replace({pd.NA: None})

    return df_filtered

def transform_dataframe(df_input_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms a wide-format DataFrame with a complex header into a long-format DataFrame,
    incorporating rules for different column types and handling dynamic metrics.

    Parameters:
    df_input_raw (pd.DataFrame): The input DataFrame, expected to have the raw
                                  structure (including header rows as data rows).

    Returns:
    pd.DataFrame: The transformed DataFrame in the desired output format,
                  with a 'Company Type' flag and dynamic columns for other metrics (excluding 'Metric Name'),
                  and with rows dropped based on specific conditions.
    """
    # Extract headers from the first two rows of the input DataFrame
    company_headers = df_input_raw.iloc[0]
    metric_headers = df_input_raw.iloc[1]

    # Actual data starts from row 2
    df_data = df_input_raw.iloc[2:].copy()
    df_data.columns = range(len(df_data.columns)) # Ensure numeric column index for data access

    output_records = []
    # To collect all unique metric names that might become dynamic columns
    all_dynamic_metric_columns = set()

    # Set of expected standard metric names for identifying blocks
    standard_metrics = {'Cal/Raw', 'Remarks', 'Marking'}
    # Define fixed output columns and their desired order (Metric Name is removed)
    fixed_output_columns = ['Field', 'Company', 'Company Type', 'Cac/Raw', 'Remarks', 'Marking']


    # Initialize variables to hold the last seen Cal/Raw and Remarks from a standard company block
    last_standard_cac_raw = None
    last_standard_remarks = None

    # Iterate through each row of the data (Field1, Field2, etc.)
    for row_idx in range(df_data.shape[0]):
        current_field = df_data.iloc[row_idx, 0] # 'FieldX' value from the first column

        col_ptr = 1 # Start scanning from the second column (index 1)

        # Reset for each new row, as previous row's 'last_standard' values don't carry over to next row
        last_standard_cac_raw = None
        last_standard_remarks = None

        while col_ptr < len(df_data.columns):
            company_h_current = company_headers[col_ptr]
            metric_h_current = metric_headers[col_ptr]

            # Initialize current record with None for all fixed columns
            current_output_record = {col: None for col in fixed_output_columns}
            current_output_record['Field'] = current_field

            company_name = None
            
            # Case 1: Standard Company block (e.g., Company1, Company2, Company3, Company4)
            # This block starts with a non-NaN company name in the first header row,
            # and 'Cal/Raw' as the metric in the second header row.
            if not pd.isna(company_h_current) and str(metric_h_current).strip() == 'Cal/Raw':
                company_name = str(company_h_current)
                current_output_record['Company'] = company_name
                current_output_record['Company Type'] = 'Y' # Flag for standard company
                # Metric Name column removed as per user request

                num_cols_processed = 0
                
                # Extract Cac/Raw
                if col_ptr < len(df_data.columns) and str(metric_headers[col_ptr]).strip() == 'Cal/Raw':
                    current_output_record['Cac/Raw'] = df_data.iloc[row_idx, col_ptr]
                    num_cols_processed += 1

                # Extract Remarks (if present and correctly identified within the same company block)
                if (col_ptr + 1) < len(df_data.columns) and \
                   (pd.isna(company_headers[col_ptr + 1]) or str(company_headers[col_ptr + 1]).strip() == company_name) and \
                   str(metric_headers[col_ptr + 1]).strip() == 'Remarks':
                    current_output_record['Remarks'] = df_data.iloc[row_idx, col_ptr + 1]
                    num_cols_processed += 1
                
                # Extract Marking (if present and correctly identified within the same company block)
                if (col_ptr + 2) < len(df_data.columns) and \
                   (pd.isna(company_headers[col_ptr + 2]) or str(company_headers[col_ptr + 2]).strip() == company_name) and \
                   str(metric_headers[col_ptr + 2]).strip() == 'Marking':
                    current_output_record['Marking'] = df_data.iloc[row_idx, col_ptr + 2]
                    num_cols_processed += 1
                
                if num_cols_processed == 0: # Fallback, though 'Cal/Raw' should usually imply at least 1 column
                    col_ptr += 1
                    continue
                
                # Update last seen standard Cac/Raw and Remarks
                last_standard_cac_raw = current_output_record['Cac/Raw']
                last_standard_remarks = current_output_record['Remarks']

                output_records.append(current_output_record)
                col_ptr += num_cols_processed 
            
            # Case 2: Standalone numeric company ID (e.g., 87688, 9861, 2189)
            # Identified by NaN in the first header row and a numeric string in the second header row.
            # Rule: Value goes to 'Marking'. Cac/Raw and Remarks inherit ONLY IF the value itself is NOT empty.
            elif pd.isna(company_h_current) and not pd.isna(metric_h_current) and str(metric_h_current).strip().isdigit():
                company_name = str(metric_h_current).strip()
                current_output_record['Company'] = company_name
                current_output_record['Company Type'] = 'Numeric' # Flag for numeric ID
                # Metric Name column removed as per user request

                current_cell_value = df_data.iloc[row_idx, col_ptr]

                # Conditional inheritance: only inherit if the cell value for the numeric ID is not empty/NaN
                if pd.isna(current_cell_value) or (isinstance(current_cell_value, str) and str(current_cell_value).strip() == ''):
                    current_output_record['Cac/Raw'] = None
                    current_output_record['Remarks'] = None
                else:
                    current_output_record['Cac/Raw'] = last_standard_cac_raw
                    current_output_record['Remarks'] = last_standard_remarks
                
                current_output_record['Marking'] = current_cell_value # The value in this single column goes to 'Marking'.

                output_records.append(current_output_record)
                col_ptr += 1 # Only one column is consumed by this type of entry

            # Case 4: Standalone NON-NUMERIC Metric ID (e.g., std_usb_subdeal)
            # Identified by NaN in the first header row and a non-numeric string in the second header row.
            elif pd.isna(company_h_current) and not pd.isna(metric_h_current) and not str(metric_h_current).strip().isdigit():
                company_name = str(metric_h_current).strip() # The metric name itself becomes the company
                current_output_record['Company'] = company_name
                current_output_record['Company Type'] = 'Non-Numeric ID' # Flag for standalone string metric
                # Metric Name column removed as per user request

                current_cell_value = df_data.iloc[row_idx, col_ptr] # Get the current cell value for inheritance check
                current_output_record['Marking'] = current_cell_value # Value goes to Marking

                # Apply conditional inheritance similar to Numeric IDs
                if pd.isna(current_cell_value) or (isinstance(current_cell_value, str) and str(current_cell_value).strip() == ''):
                    current_output_record['Cac/Raw'] = None
                    current_output_record['Remarks'] = None
                else:
                    current_output_record['Cac/Raw'] = last_standard_cac_raw # Inherit if not empty
                    current_output_record['Remarks'] = last_standard_remarks # Inherit if not empty
                
                output_records.append(current_output_record)
                col_ptr += 1 # Only one column is consumed by this type of entry

            # Case 3: Non-standard metric with a company header (e.g., (Company5, 'OtherMetric'))
            # Applies if first header is not NaN, and second header is not a standard metric
            # AND not a numeric digit.
            # Rule: Value goes to a new column named after the metric. Cac/Raw, Remarks, Marking are None.
            elif not pd.isna(company_h_current) and \
                 not pd.isna(metric_h_current) and \
                 str(metric_h_current).strip() not in standard_metrics and \
                 not str(metric_h_current).strip().isdigit():
                
                company_name = str(company_h_current)
                metric_name = str(metric_h_current).strip()
                cell_value = df_data.iloc[row_idx, col_ptr] # Get the value from the cell
                
                current_output_record['Company'] = company_name
                current_output_record['Company Type'] = 'Other Metric' # Flag for non-standard metric
                # Metric Name column removed as per user request
                
                # Add the dynamic metric name to our set of all possible columns
                all_dynamic_metric_columns.add(metric_name)
                
                # Assign the value to the dynamically named column
                current_output_record[metric_name] = cell_value

                # Cac/Raw, Remarks, Marking are None for these entries as value is in its own column
                current_output_record['Cac/Raw'] = None
                current_output_record['Remarks'] = None
                current_output_record['Marking'] = None

                output_records.append(current_output_record)
                col_ptr += 1 # Only one column is consumed by this type of entry

            else:
                # If a column doesn't fit any of the above patterns (e.g., it's a metric for a company block
                # that has already been processed in a previous iteration, or an unhandled format like
                # a completely blank column), simply advance the pointer to the next column.
                col_ptr += 1

    df_output = pd.DataFrame(output_records)

    # Determine the final column order
    # Sort dynamic columns alphabetically for consistent output
    sorted_dynamic_metrics = sorted(list(all_dynamic_metric_columns))
    final_columns_order = fixed_output_columns + sorted_dynamic_metrics
    
    # Filter to only columns that actually appeared in the data (as some might not be in all records)
    final_columns_order = [col for col in final_columns_order if col in df_output.columns]

    df_output = df_output[final_columns_order]

    # Replace pandas NA values, empty strings, and strings with only spaces with None for consistency.
    df_output = df_output.replace({pd.NA: None, '': None, ' ': None})
    
    # Apply the row dropping function
    final_df = drop_rows_based_on_conditions(df_output)

    return final_df

# --- Example Usage (How you would use the function) ---
# IMPORTANT: Ensure 'openpyxl' is installed in your environment for pandas to read .xlsx files:
# Open your terminal or command prompt and run:
# pip install openpyxl

# Load your input DataFrame from the Excel file.
# Make sure "final_input.xlsx" (or "usecase2.xlsx" if that's your test file)
# is in the same directory as your script, or provide the full path to the file.
try:
    my_input_dataframe = pd.read_excel("final_input2.xlsx", header=None) # Using final_input.xlsx as per your example

    # Call the transformation function with your DataFrame
    transformed_output_dataframe = transform_dataframe(my_input_dataframe)

    # Write the final output DataFrame to an Excel file
    output_excel_filename = "testing_output.xlsx" # Consistent output filename
    transformed_output_dataframe.to_excel(output_excel_filename, index=False)

    print(f"Transformation complete. Output saved to '{output_excel_filename}'")
    print("\nSample of the transformed DataFrame (first 10 rows):")
    print(transformed_output_dataframe.head(10).to_string())

except FileNotFoundError:
    print("Error: The input Excel file was not found.")
    print("Please ensure 'final_input.xlsx' (or your designated input file) is in the same directory as your script, or provide the full path.")
except Exception as e:
    if "No such keys(s): 'io.excel.zip.reader'" in str(e):
        print(f"An error occurred: {e}")
        print("\nThis error typically means the 'openpyxl' library is not correctly installed or accessible.")
        print("Please ensure you have installed it by running 'pip install openpyxl' in your terminal,")
        print("and that you are running the script in the same Python environment where it was installed.")
    else:
        print(f"An unexpected error occurred: {e}")