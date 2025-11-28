import csv
import sys
import os

def parse_percentiles_max(value_str):
    """
    Parses the percentiles string (e.g., '[1.0, 2.0, ...]') and returns the maximum value.
    Assumes the last value in the list is the maximum.
    """
    try:
        clean_str = value_str.strip('[]')
        if not clean_str:
            return None
        values = [float(x.strip()) for x in clean_str.split(',')]
        return max(values)
    except Exception:
        return None

def extract_metrics(csv_path):
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return

    # Use utf-8-sig to handle BOM if present
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        try:
            # Read 3 header lines
            header1 = next(reader)
            header2 = next(reader)
            header3 = next(reader)
        except StopIteration:
            print(f"Error: CSV file {csv_path} is empty or malformed.")
            return

        # Create a mapping from (Level1, Level2, Level3) header to column index
        col_map = {}
        for i, (h1, h2, h3) in enumerate(zip(header1, header2, header3)):
            # Normalize keys: strip whitespace
            key = (h1.strip(), h2.strip(), h3.strip())
            col_map[key] = i
        
        data_rows = list(reader)
        
        # Define the metrics we want to extract
        metrics_config = [
            ("Output Throughput", "Token Throughput", "Successful Output Tokens/Sec"),
            ("TTFT", "Time to First Token", "Successful ms"),
            ("TPOT", "Time per Output Token", "Successful ms"),
            ("ITL", "Inter Token Latency", "Successful ms")
        ]

        # Prepare Headers
        headers = ["Output Throughput (Mean)"]
        for name, _, _ in metrics_config[1:]:
             headers.append(f"{name} (Median)")
             headers.append(f"{name} (Mean)")
             headers.append(f"{name} (Max)")
        
        # Collect all rows data
        all_output_rows = []

        for row_idx, row in enumerate(data_rows):
            if not row: continue  # Skip empty lines
            
            # Helper to safely get value from row
            def get_val(l1, l2, l3):
                key = (l1, l2, l3)
                idx = col_map.get(key)
                if idx is not None and idx < len(row):
                    val = row[idx].strip()
                    if val == "": return None
                    return val
                return None
            
            row_values = []

            # 1. Output Throughput (Mean only)
            l1, l2 = metrics_config[0][1], metrics_config[0][2]
            avg = get_val(l1, l2, "Mean")
            row_values.append(f"{float(avg):.3f}" if avg else "N/A")

            # 2. Others (Median, Mean, Max)
            for name, l1, l2 in metrics_config[1:]:
                median = get_val(l1, l2, "Median")
                avg = get_val(l1, l2, "Mean")
                percentiles = get_val(l1, l2, "Percentiles")
                
                max_val = "N/A"
                if percentiles:
                    m = parse_percentiles_max(percentiles)
                    if m is not None:
                        max_val = f"{m:.3f}"
                
                row_values.append(f"{float(median):.3f}" if median else "N/A")
                row_values.append(f"{float(avg):.3f}" if avg else "N/A")
                row_values.append(max_val)
            
            all_output_rows.append(row_values)

        # Calculate column widths
        col_widths = [len(h) for h in headers]
        for r in all_output_rows:
            for i, val in enumerate(r):
                col_widths[i] = max(col_widths[i], len(val))

        # Print Header
        header_fmt = [f"{h:<{col_widths[i]}}" for i, h in enumerate(headers)]
        print(" , ".join(header_fmt))

        # Print Rows
        for r in all_output_rows:
            row_fmt = [f"{val:<{col_widths[i]}}" for i, val in enumerate(r)]
            print(" , ".join(row_fmt))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_metrics.py <csv_file>")
        sys.exit(1)
    
    extract_metrics(sys.argv[1])
