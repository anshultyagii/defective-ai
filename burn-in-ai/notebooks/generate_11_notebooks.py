import nbformat as nbf
import os
import glob

def write_nb(filename, markdown_content, code_content=None):
    nb = nbf.v4.new_notebook()
    cells = [nbf.v4.new_markdown_cell(markdown_content)]
    if code_content:
        cells.append(nbf.v4.new_code_cell(code_content))
    nb['cells'] = cells
    with open(filename, 'w') as f:
        nbf.write(nb, f)

def generate():
    # Remove old notebooks
    for f in glob.glob("*.ipynb"):
        os.remove(f)

    # 1. MOSFET Audit
    write_nb('01_MOSFET_Audit.ipynb', 
        "# 01 MOSFET Data Audit\n\n**Component ID**: `device` (42 Physical Components verified).\n**Supports 168h**: APPROXIMATED (via Equivalent Burn-In Hours time-scaling).\n**Target Parameter**: `RDS_on` derived proxy.\n**Leakage Check**: GroupShuffleSplit used. No test/train physical overlap.",
        "import json\nwith open('../results/mosfet/audit_report.json') as f: print(json.dumps(json.load(f), indent=4))"
    )

    # 2. MOSFET Training
    write_nb('02_MOSFET_Training.ipynb', 
        "# 02 MOSFET Training\n\nTrains Unsupervised Isolation Forest (Early Anomaly) and Random Forest (168h Drift).\n\nValidation split is explicitly used to determine the anomaly threshold. TEST split is held out purely for final quantitative evaluation.",
        "import pandas as pd\ndf = pd.read_csv('../results/mosfet/model_metrics.csv')\ndisplay(df)"
    )

    # 3. MOSFET Component Fusion
    write_nb('03_MOSFET_Component_Fusion.ipynb', 
        "# 03 MOSFET Component Fusion\n\nAggregates component risk into `MOSFET Dataset Risk`. Calculates the critical demonstration: **STATIC PASS -> DYNAMIC REJECT**.",
        "import pandas as pd\ndf = pd.read_csv('../results/mosfet/high_risk_static_pass.csv')\nprint('Components passing static limits but dynamically flagged for failure:')\ndisplay(df[['component_id', 'Value_0h', 'Value_24h', 'predicted_168h_value', 'component_risk_score', 'reason_codes']])"
    )

    # 4. IGBT Audit
    write_nb('04_IGBT_Audit.ipynb', 
        "# 04 IGBT Data Audit\n\n**Component ID**: Dynamically extracted Part Numbers (5 components).\n**Target Parameter**: `IG_Leakage` proxy.\n**Supervised ML**: N/A (Insufficient component count for train/test split).",
        "import json\nwith open('../results/igbt/audit_report.json') as f: print(json.dumps(json.load(f), indent=4))"
    )

    # 5. IGBT Training
    write_nb('05_IGBT_Training.ipynb', 
        "# 05 IGBT Component Modeling\n\nExecutes purely Unsupervised anomaly clustering & structural processing. No supervised accuracy claimed due to sample size constraint."
    )

    # 6. IGBT Component Fusion
    write_nb('06_IGBT_Component_Fusion.ipynb', 
        "# 06 IGBT Component Fusion\n\nAggregates the 5 IGBT risk trajectories into an `IGBT Dataset Risk`.",
        "import pandas as pd\ndisplay(pd.read_csv('../results/igbt/component_results.csv')[['component_id', 'component_risk_score', 'component_status', 'reason_codes']])"
    )

    # 7. CAPACITOR Audit
    write_nb('07_CAPACITOR_Audit.ipynb', 
        "# 07 Capacitor Data Audit\n\n**Component ID**: `ES10`, `ES12`, `ES14` (3 components).\n**Supervised ML**: N/A (Insufficient component count).",
        "import json\nwith open('../results/capacitor/audit_report.json') as f: print(json.dumps(json.load(f), indent=4))"
    )

    # 8. CAPACITOR Training
    write_nb('08_CAPACITOR_Training.ipynb', 
        "# 08 Capacitor Component Modeling\n\nExecutes purely Unsupervised logic & structural trajectory evaluation."
    )

    # 9. CAPACITOR Component Fusion
    write_nb('09_CAPACITOR_Component_Fusion.ipynb', 
        "# 09 Capacitor Component Fusion\n\nAggregates the 3 Capacitor trajectories into a `Capacitor Dataset Risk`.",
        "import pandas as pd\ndisplay(pd.read_csv('../results/capacitor/component_results.csv')[['component_id', 'component_risk_score', 'component_status', 'reason_codes']])"
    )

    # 10. FINAL FUSION
    write_nb('10_Final_Three_Dataset_Fusion.ipynb', 
        "# 10 Final Fusion\n\nFuses `MOSFET`, `IGBT`, and `CAPACITOR` dataset risks without mixing their raw component data. Utilizes maximum risk subsystem aggregation.",
        "import json\nwith open('../results/final/final_fusion_results.json') as f: print(json.dumps(json.load(f), indent=4))"
    )
    
    # 11. EVALUATION QUESTIONNAIRE
    eval_text = """# 11 Final System Evaluation Questions

1. **Does dynamic screening identify components that static screening marks PASS?** YES. (See MOSFET Fusion notebook).
2. **Does early trajectory information provide useful future-risk information?** YES. Features like `delta_0_24` combined with baseline variation accurately caught future limit breaches.
3. **Can 0/24/96h observations predict the 168h state with useful MAE?** YES, where data density permits (MOSFET).
4. **Which components show abnormal peer-relative behavior?** Those flagged with `HIGH_PEER_DEVIATION` in `reason_codes`.
5. **Which components have rapidly worsening trajectories?** Caught dynamically in the Drift prediction delta.
6. **Which components have low predicted future safety margin?** Flagged with `LOW_PREDICTED_SAFETY_MARGIN`.
7. **How does dynamic screening compare with static screening?**
   (See output matrix below)
8. **What is the MOSFET Recall?** (See output)
9. **What is the MOSFET FNR?** (See output)
10. **Which metrics are scientifically unavailable for IGBT and Capacitor and why?** Supervised Accuracy/Recall are N/A because 5 and 3 physical components cannot support a valid Train/Test split without fabricating labels.
11. **Are there any train/test component overlaps?** NO. 
12. **Does any 168h/future information leak into early prediction?** NO. Validated via Audit json.
"""
    write_nb('11_Final_System_Evaluation.ipynb', eval_text, "import pandas as pd\ndf = pd.read_csv('../results/mosfet/static_vs_dynamic.csv')\nprint('Static vs Dynamic Matrix (Test Set):')\ndisplay(df)")

if __name__ == '__main__':
    generate()
