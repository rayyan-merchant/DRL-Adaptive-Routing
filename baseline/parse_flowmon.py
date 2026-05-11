import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import os
import sys
import re

def parse_flowmon(xml_file, duration=99.0):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    rows = []
    
    for flow in root.iter("Flow"):
        flow_id = flow.get("flowId")
        tx = int(flow.get("txPackets", 0))
        rx = int(flow.get("rxPackets", 0))
        loss_ratio = (tx - rx) / tx if tx > 0 else 0.0
        
        # Handle delaySum: strip 'ns', '+' and anything else
        delay_sum_str = flow.get("delaySum", "0ns")
        delay_sum_str = delay_sum_str.replace("ns", "").replace("+", "")
        # Remove any remaining non-numeric characters except '.'
        delay_sum_str = re.sub(r'[^\d.]', '', delay_sum_str)
        if not delay_sum_str:
            delay_sum_str = "0"
            
        delay_sum_ns = float(delay_sum_str)
        delay_sum_ms = delay_sum_ns / 1e6
        avg_delay_ms = delay_sum_ms / rx if rx > 0 else 0.0
        
        rx_bytes = int(flow.get("rxBytes", 0))
        throughput_mbps = (rx_bytes * 8) / (duration * 1e6)
        
        rows.append({
            "flow_id": flow_id,
            "throughput_mbps": throughput_mbps,
            "avg_delay_ms": avg_delay_ms,
            "loss_ratio": loss_ratio
        })
        
    return pd.DataFrame(rows)

def summarize(df):
    if df.empty:
        return {"throughput": 0.0, "delay": 0.0, "loss": 0.0}
    return {
        "throughput": df["throughput_mbps"].mean(),
        "delay": df["avg_delay_ms"].mean(),
        "loss": df["loss_ratio"].mean()
    }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python parse_flowmon.py <xml_file>")
        sys.exit(1)
        
    xml_path = sys.argv[1]
    df = parse_flowmon(xml_path)
    
    print(df.to_string(index=False))
    print(summarize(df))
    
    # Save to results/logs/
    os.makedirs("../results/logs", exist_ok=True)
    base_name = os.path.basename(xml_path)
    csv_name = os.path.splitext(base_name)[0] + ".csv"
    out_path = os.path.join("../results/logs", csv_name)
    df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")
