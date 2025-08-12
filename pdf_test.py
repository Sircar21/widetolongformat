import camelot
import re
from rapidfuzz import fuzz

# Regex to match numbers with commas, decimals, percentages
NUM_RE = re.compile(r"[-]?\(?\d[\d,]*(?:\.\d+)?%?\)?")
FUZZY_THRESHOLD = 80

DATA_POINTS = {}

def is_year_like(val):
    """Avoid picking years like 2006."""
    try:
        v = int(str(val).replace(",", "").replace("(", "").replace(")", ""))
        return 1900 <= v <= 2100
    except:
        return False

def normalize_num(text):
    """Convert cleaned string to float."""
    if not text:
        return None
    sign = -1 if (text.startswith("(") and text.endswith(")")) else 1
    txt = text.strip("()").replace(",", "").replace("%", "")
    try:
        return float(txt) * sign
    except:
        return None

def extract_datapoints(pdf_path, datapoints):
    results = {k: None for k in datapoints}
    
    # Extract all tables using hybrid method
    tables = camelot.read_pdf(pdf_path, flavor="hybrid", pages="all")
    
    for table in tables:
        df = table.df.fillna("").astype(str)
        
        # Loop over each cell in the table
        for i in range(len(df)):
            for j in range(len(df.columns)):
                cell = df.iloc[i, j].strip()
                for key, patterns in datapoints.items():
                    for pat in patterns:
                        if pat.lower() in cell.lower() or fuzz.partial_ratio(pat.lower(), cell.lower()) >= FUZZY_THRESHOLD:
                            
                            # Check right cell for value
                            if j + 1 < len(df.columns):
                                val = df.iloc[i, j+1].strip()
                                if NUM_RE.search(val) and not is_year_like(val):
                                    results[key] = normalize_num(NUM_RE.search(val).group())
                                    break
                            
                            # If not found, check below cell
                            if results[key] is None and i + 1 < len(df):
                                val = df.iloc[i+1, j].strip()
                                if NUM_RE.search(val) and not is_year_like(val):
                                    results[key] = normalize_num(NUM_RE.search(val).group())
                                    break
    
    return results

if __name__ == "__main__":
    pdf_file = "test.pdf"  # replace with your PDF
    extracted = extract_datapoints(pdf_file, DATA_POINTS)
    for k, v in extracted.items():
        print(f"{k}: {v}")
