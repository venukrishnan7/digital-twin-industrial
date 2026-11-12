import csv
import time
from datetime import datetime
from pathlib import Path

OUTPUT_FILE = Path(__file__).resolve().parent / "factory_twin_phase5b_data.csv"


class DigitalTwin:
    """Rule-based twin for the simulated sequential factory process."""

    def predict(self, state, drill_timer, drill_delay, sensor_wait=0):
        expected = {
            "part_detected": False,
            "clamp_closed": False,
            "drill_complete": False,
        }

        if state in ("CLAMPING", "DRILLING", "UNCLAMPING", "RELEASING", "COMPLETE"):
            expected["part_detected"] = True
        elif state == "CONVEYING" and sensor_wait >= 1:
            # Once the conveyor has had a chance to present a part,
            # a missing detection signal is a mismatch.
            expected["part_detected"] = True

        if state in ("DRILLING", "UNCLAMPING"):
            expected["clamp_closed"] = True

        if state in ("UNCLAMPING", "RELEASING", "COMPLETE"):
            expected["drill_complete"] = True

        if state == "DRILLING" and drill_timer >= drill_delay:
            expected["drill_complete"] = True

        # In FAULT, the normal process-phase sensor expectations are no
        # longer reliable. Return unavailable predictions rather than
        # inventing values that could create false sensor mismatches.
        if state == "FAULT":
            return None
        return expected

    @staticmethod
    def compare(observed, expected):
        return {
            name: int(observed[name]) - int(expected[name])
            for name in expected
        }


class FactorySimulation:
    def __init__(self, mode, run_id):
        self.mode = mode
        self.run_id = run_id
        self.state = "IDLE"
        self.cycle = 0
        self.drill_timer = 0
        self.sensor_wait = 0
        self.drill_delay = 5 if mode == 2 else 2
        self.max_drill_time = 3
        self.max_sensor_wait = 5
        self.product_count = 0
        self.completed = False
        self.twin = DigitalTwin()
        self.rows = []
        self.faults = set()
        self.sensors = {
            "part_detected": False,
            "clamp_closed": False,
            "drill_complete": False,
        }
        self.actuators = {"conveyor": False, "clamp": False, "drill": False}

    def add_fault(self, fault_type, message):
        self.faults.add(fault_type)
        print(f"FAULT DETECTED: {message}")

    def validate_safety(self):
        if self.actuators["drill"] and not self.sensors["clamp_closed"]:
            self.add_fault("clamp_interlock", "Drilling requested without clamp confirmation")
            self.actuators["drill"] = False
            return False
        return True

    def step(self):
        self.cycle += 1
        state_before = self.state
        timing_violation = 0
        sensor_timeout = 0
        safety_violation = 0

        if self.state == "IDLE":
            self.actuators["conveyor"] = True
            self.state = "CONVEYING"

        elif self.state == "CONVEYING":
            self.sensor_wait += 1
            if self.mode != 3:
                self.sensors["part_detected"] = True

            if self.sensors["part_detected"]:
                self.actuators["conveyor"] = False
                self.state = "CLAMPING"
            elif self.sensor_wait >= self.max_sensor_wait:
                sensor_timeout = 1
                self.add_fault("part_sensor_failure", "Part sensor timeout")
                self.state = "FAULT"

        elif self.state == "CLAMPING":
            self.actuators["clamp"] = True
            if self.mode == 4:
                self.actuators["clamp"] = False
                self.sensors["clamp_closed"] = False
            else:
                self.sensors["clamp_closed"] = True
            self.state = "DRILLING"

        elif self.state == "DRILLING":
            if not self.sensors["clamp_closed"]:
                safety_violation = 1
                self.add_fault("clamp_interlock", "Clamp is not confirmed closed")
                self.actuators["drill"] = False
                self.state = "FAULT"
            else:
                self.actuators["drill"] = True
                if not self.validate_safety():
                    safety_violation = 1
                    self.state = "FAULT"
                else:
                    self.drill_timer += 1
                    if self.drill_timer > self.max_drill_time:
                        timing_violation = 1
                        self.add_fault("slow_drilling", "Drilling exceeded allowed cycle time")
                    if self.drill_timer >= self.drill_delay:
                        self.actuators["drill"] = False
                        self.sensors["drill_complete"] = True
                        self.state = "UNCLAMPING"

        elif self.state == "UNCLAMPING":
            self.actuators["clamp"] = False
            self.sensors["clamp_closed"] = False
            self.state = "RELEASING"

        elif self.state == "RELEASING":
            self.actuators["conveyor"] = True
            self.state = "COMPLETE"

        elif self.state == "COMPLETE":
            self.actuators["conveyor"] = False
            self.product_count += 1
            self.completed = True
            self.record(state_before, 0, 0, 0)
            return True

        elif self.state == "FAULT":
            self.actuators = {"conveyor": False, "clamp": False, "drill": False}
            self.record(state_before, 0, 0, 0)
            return True

        self.record(state_before, timing_violation, sensor_timeout, safety_violation)
        return False

    def record(self, state_before, timing_violation, sensor_timeout, safety_violation):
        expected = self.twin.predict(
            self.state, self.drill_timer, self.drill_delay, self.sensor_wait
        )
        prediction_valid = int(expected is not None)
        if prediction_valid:
            residuals = self.twin.compare(self.sensors, expected)
            residual_count = sum(value != 0 for value in residuals.values())
        else:
            # Sensor residuals are not evaluated in FAULT because normal
            # process expectations are undefined in that state.
            residuals = None
            residual_count = 0
        # Separate three concepts:
        # 1) residual_mismatch: observed sensors differ from twin predictions now
        # 2) fault_event: a new mismatch or rule trigger occurs on this row
        # 3) fault_active: a fault has been detected and remains active
        residual_mismatch = int(prediction_valid and residual_count > 0)
        rule_event = int(bool(timing_violation or sensor_timeout or safety_violation))
        fault_event = int(bool(residual_mismatch or rule_event))

        # The simulation's fault set is cumulative, so this captures whether
        # a detected scenario fault remains active after its trigger row.
        fault_active = int(bool(self.faults) or residual_mismatch)
        row_fault_types = sorted(self.faults)
        if residual_mismatch:
            mismatch_types = []
            if residuals["part_detected"] != 0:
                mismatch_types.append("part_sensor_residual")
            if residuals["clamp_closed"] != 0:
                mismatch_types.append("clamp_sensor_residual")
            if residuals["drill_complete"] != 0:
                mismatch_types.append("drill_completion_residual")
            # Include the mismatch category in the row-level type without
            # removing any active scenario fault category.
            row_fault_types = sorted(set(row_fault_types + mismatch_types))
        row_fault = fault_active

        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "run_id": self.run_id,
            "mode": self.mode,
            "cycle": self.cycle,
            "state_before": state_before,
            "machine_state": self.state,
            "drill_timer": self.drill_timer,
            "sensor_wait": self.sensor_wait,
            "part_detected": int(self.sensors["part_detected"]),
            "clamp_closed": int(self.sensors["clamp_closed"]),
            "drill_complete": int(self.sensors["drill_complete"]),
            "prediction_valid": prediction_valid,
            "predicted_part": int(expected["part_detected"]) if prediction_valid else "",
            "predicted_clamp": int(expected["clamp_closed"]) if prediction_valid else "",
            "predicted_drill_complete": int(expected["drill_complete"]) if prediction_valid else "",
            "residual_part": residuals["part_detected"] if prediction_valid else "",
            "residual_clamp": residuals["clamp_closed"] if prediction_valid else "",
            "residual_drill_complete": residuals["drill_complete"] if prediction_valid else "",
            "residual_count": residual_count,
            "residual_mismatch": residual_mismatch,
            "timing_violation": int(timing_violation),
            "sensor_timeout": int(sensor_timeout),
            "safety_violation": int(safety_violation),
            "rule_event": rule_event,
            "fault_event": fault_event,
            "fault_active": fault_active,
            # Backward-compatible label: equivalent to fault_active.
            "fault_label": row_fault,
            "fault_type": "|".join(row_fault_types) if row_fault_types else "normal",
            "actuator_conveyor": int(self.actuators["conveyor"]),
            "actuator_clamp": int(self.actuators["clamp"]),
            "actuator_drill": int(self.actuators["drill"]),
        }
        self.rows.append(row)
        print(
            f"Cycle {self.cycle:02d} | {self.state:12s} | "
            f"residuals={residuals if prediction_valid else 'not evaluated (FAULT state)'} | "
            f"fault={row['fault_type']}"
        )

    def run(self):
        print(f"\n--- Run {self.run_id}: mode {self.mode} ---")
        for _ in range(20):
            if self.step():
                break
            time.sleep(0.05)
        return self.rows


def main():
    print("DIGITAL TWIN - PHASE 5B DATA COLLECTION")
    print("This creates a fresh CSV with separate mismatch, event, and active-fault labels.")
    print("1. Normal operation")
    print("2. Slow drilling")
    print("3. Part sensor failure")
    print("4. Clamp safety interlock")
    all_rows = []
    descriptions = {
        1: "normal",
        2: "slow_drilling",
        3: "part_sensor_failure",
        4: "clamp_interlock",
    }

    for mode in (1, 2, 3, 4):
        simulation = FactorySimulation(mode, run_id=f"mode_{mode}")
        all_rows.extend(simulation.run())

    if not all_rows:
        print("No records generated.")
        return

    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    print("\n===== DATA COLLECTION SUMMARY =====")
    print(f"Saved {len(all_rows)} rows to: {OUTPUT_FILE}")
    print("Fault types by mode:")
    for mode, name in descriptions.items():
        rows = [r for r in all_rows if r["mode"] == mode]
        positive = sum(r["fault_label"] for r in rows)
        print(f"Mode {mode} ({name}): {len(rows)} rows; fault-labelled rows={positive}")
    print("\nThis CSV is for rule-validation analysis, not sufficient by itself to train a robust Random Forest.\nNew columns: residual_mismatch, rule_event, fault_event, fault_active.\nfault_label is retained for compatibility and equals fault_active.")


if __name__ == "__main__":
    main()
