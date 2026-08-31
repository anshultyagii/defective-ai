# NASA Power MOSFET Thermal Overstress Aging Datasets (Devices 01 – 42)

This directory contains separate clean, continuous time-series CSV datasets for all **42 power MOSFET components** tested in the **NASA Ames Prognostics Center of Excellence (PCoE)** Power MOSFET Thermal Overstress Aging study.

---

## 1. Directory Structure

```
MOSFET_Individual_Device_Datasets/
├── 00_Master_MOSFET_Device_Inventory.csv   # Master summary manifest across all 42 devices
├── MOSFET_Device_01.csv                    # Device 01 continuous time series
├── MOSFET_Device_02.csv                    # Device 02 continuous time series
...
├── MOSFET_Device_42.csv                    # Device 42 continuous time series
└── README.md                               # Dataset documentation & data dictionary
```

---

## 2. Data Dictionary (`MOSFET_Device_XX.csv`)

| Column Name | Units | Data Type | Description |
| :--- | :---: | :---: | :--- |
| `Device_ID` | — | Integer | Identifier for the MOSFET device under test (1 to 42). |
| `Run_ID` | — | Integer | Measurement run / thermal stress session index for this device. |
| `Timestamp` | — | String | Original timestamp from telemetry data (`MM/DD/YYYY HH:MM:SS.fff`). |
| `Datetime` | — | ISO 8601 | Standard parsed datetime (`YYYY-MM-DD HH:MM:SS.fff`). |
| `Aging_Hours` | Hours | Float | Cumulative test time in hours elapsed since the start of Run 1. |
| `Run_Elapsed_Seconds` | Seconds | Float | Elapsed time within the current individual run session. |
| `Supply_Voltage_V` | Volts | Float | Power supply DC rail voltage ($V_{\text{supply}}$). |
| `Package_Temp_C` | °C | Float | Measured MOSFET package / case temperature ($T_{\text{package}}$). |
| `Flange_Temp_C` | °C | Float | Measured heatsink / flange temperature ($T_{\text{flange}}$). |
| `Delta_Temp_C` | °C | Float | Temperature differential: $T_{\text{package}} - T_{\text{flange}}$ (Thermal impedance indicator). |
| `VDS_V` | Volts | Float | Drain-to-Source voltage ($V_{DS}$). |
| `Current_A` | Amperes | Float | Drain conduction current ($I_D$). |
| `RDS_on_Ohm` | $\Omega$ | Float | On-state resistance: $R_{DS(on)} = V_{DS} / I_D$ (computed during conduction). |
| `Baseline_RDS_Ohm` | $\Omega$ | Float | Initial pristine on-resistance measured at start of device life. |
| `RDS_Degradation_Pct`| % | Float | Percentage degradation relative to baseline: $100 \times (R_{DS} - R_{DS,\text{base}}) / R_{DS,\text{base}}$. |
| `Gate_Voltage_Setpoint_V` | Volts | Float | Gate driver PWM control voltage setpoint (e.g. 10 V). |
| `Supply_Voltage_Setpoint_V`| Volts | Float | Power supply setpoint (e.g. 5 V / 8 V). |
| `Low_Temp_Setpoint_C` | °C | Float | PWM temperature controller low-temperature threshold. |
| `High_Temp_Setpoint_C` | °C | Float | PWM temperature controller high-temperature threshold. |
| `Shutdown_Temp_Setpoint_C` | °C | Float | Safety shutdown temperature limit (e.g. 280 °C). |
| `Switching_Frequency_Hz` | Hz | Float | PWM switching frequency (1,000 Hz nominal). |
| `Duty_Cycle_Pct` | % | Float | PWM duty cycle percentage (40% nominal). |

---

## 3. Master Inventory (`00_Master_MOSFET_Device_Inventory.csv`)

The master inventory provides device-level prognostic metrics:
- **`Aging_Category`**:
  - `TREND_ELIGIBLE`: Devices with $\ge 3$ aging runs suitable for continuous degradation modeling & RUL estimation (e.g. Devices 8, 9, 10, 11, 12, 13, 14, 25, 27, 34).
  - `TWO_POINT_ONLY`: Devices with 2 runs (early failure / truncated test).
  - `BASELINE_ONLY`: Devices with 1 single baseline measurement run.
- **`Failure_Status`**:
  - `SEVERE_DEGRADATION`: $>50\%$ increase in $R_{DS(on)}$ or catastrophic failure.
  - `MODERATE_DEGRADATION`: $20\% - 50\%$ increase in $R_{DS(on)}$.
  - `NOMINAL_AGING`: $<20\%$ change in $R_{DS(on)}$.
  - `BASELINE_ONLY`: Pristine / single run without degradation progression.

---

## 4. Key Failure Mechanisms Studied

1. **Die-Attach & Solder Fatigue**: Increases thermal resistance ($R_{th}$), visible through escalating `Delta_Temp_C`.
2. **Bond Wire Lift-Off & Gate Degradation**: Causes sharp or progressive escalations in `RDS_on_Ohm` and `RDS_Degradation_Pct`.
3. **Thermal Runaway**: Leads to open-circuit or short-circuit catastrophic breakdown (as seen in Devices 19 and 25).
