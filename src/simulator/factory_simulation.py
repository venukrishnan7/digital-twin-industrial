import csv
import os
import time
from datetime import datetime


class DigitalTwin:
    """Rule-based Digital Twin that predicts expected sensor states."""

    def __init__(self):
        self.predictions = {
            "part_detected": False,
            "clamp_closed": False,
            "drill_complete": False
        }

    def predict(self, state, drill_timer, drill_delay):
        expected = {
            "part_detected": False,
            "clamp_closed": False,
            "drill_complete": False
        }

        if state in (
            "CLAMPING", "DRILLING", "UNCLAMPING",
            "RELEASING", "COMPLETE"
        ):
            expected["part_detected"] = True

        if state in ("DRILLING", "UNCLAMPING"):
            expected["clamp_closed"] = True

        if state in ("UNCLAMPING", "RELEASING", "COMPLETE"):
            expected["drill_complete"] = True

        if state == "DRILLING" and drill_timer >= drill_delay:
            expected["drill_complete"] = True

        self.predictions = expected
        return expected

    def compare(self, observed, expected):
        residuals = {}
        for sensor in expected:
            residuals[sensor] = int(observed[sensor]) - int(expected[sensor])
        return residuals


class FactorySimulation:
    def __init__(self, mode):
        self.mode = mode
        self.state = "IDLE"
        self.cycle = 0
        self.drill_timer = 0
        self.sensor_wait = 0
        self.drill_delay = 5 if mode == 2 else 2
        self.max_drill_time = 3
        self.max_sensor_wait = 5
        self.product_count = 0
        self.faults = []
        self.completed = False
        self.twin = DigitalTwin()
        self.mismatch_count = 0

        # Rows collected during this run are appended to the CSV at the end.
        self.dataset_rows = []
        self.csv_file = "factory_twin_data.csv"

        self.sensors = {
            "part_detected": False,
            "clamp_closed": False,
            "drill_complete": False
        }

        self.actuators = {
            "conveyor": False,
            "clamp": False,
            "drill": False
        }

    def log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}")

    def report_fault(self, message):
        if message not in self.faults:
            self.faults.append(message)
            self.log("FAULT DETECTED: " + message)

    def validate_safety(self):
        if self.actuators["drill"] and not self.sensors["clamp_closed"]:
            self.report_fault(
                "Safety violation: drilling requested without clamp confirmation."
            )
            self.actuators["drill"] = False
            return False
        return True

    def validate_twin(self):
        expected = self.twin.predict(
            self.state, self.drill_timer, self.drill_delay
        )
        residuals = self.twin.compare(self.sensors, expected)

        print("\n--- DIGITAL TWIN COMPARISON ---")
        print(f"Machine state : {self.state}")
        print(f"Expected      : {expected}")
        print(f"Observed      : {self.sensors}")
        print(f"Residuals     : {residuals}")

        mismatches = [
            name for name, value in residuals.items() if value != 0
        ]

        if mismatches:
            self.mismatch_count += len(mismatches)
            self.log("TWIN MISMATCH: " + ", ".join(mismatches))
        else:
            self.log("Twin comparison passed: all sensor states match.")

        # Store one row for each Twin comparison.
        self.dataset_rows.append({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "mode": self.mode,
            "cycle": self.cycle,
            "machine_state": self.state,
            "drill_timer": self.drill_timer,
            "part_detected": int(self.sensors["part_detected"]),
            "clamp_closed": int(self.sensors["clamp_closed"]),
            "drill_complete": int(self.sensors["drill_complete"]),
            "predicted_part": int(expected["part_detected"]),
            "predicted_clamp": int(expected["clamp_closed"]),
            "predicted_drill_complete": int(expected["drill_complete"]),
            "residual_part": residuals["part_detected"],
            "residual_clamp": residuals["clamp_closed"],
            "residual_drill_complete": residuals["drill_complete"],
            "fault_label": int(bool(self.faults)),
            "actuator_conveyor": int(self.actuators["conveyor"]),
            "actuator_clamp": int(self.actuators["clamp"]),
            "actuator_drill": int(self.actuators["drill"])
        })

        print("-------------------------------\n")

    def step(self):
        self.cycle += 1
        self.log(f"Cycle {self.cycle}: {self.state}")

        if self.state == "IDLE":
            self.actuators["conveyor"] = True
            self.log("Conveyor started.")
            self.state = "CONVEYING"

        elif self.state == "CONVEYING":
            self.sensor_wait += 1

            if self.mode != 3:
                self.sensors["part_detected"] = True

            if self.sensors["part_detected"]:
                self.actuators["conveyor"] = False
                self.log("Part detected. Conveyor stopped.")
                self.state = "CLAMPING"

            elif self.sensor_wait >= self.max_sensor_wait:
                self.report_fault(
                    "Sensor timeout: part not detected within "
                    f"{self.max_sensor_wait} cycles."
                )
                self.state = "FAULT"

            else:
                self.log("Waiting for part detection.")

        elif self.state == "CLAMPING":
            self.actuators["clamp"] = True

            if self.mode == 4:
                # Inject a clamp failure to test the safety interlock.
                self.actuators["clamp"] = False
                self.sensors["clamp_closed"] = False
            else:
                self.sensors["clamp_closed"] = True

            self.log(f"Clamp confirmation: {self.sensors['clamp_closed']}")
            self.state = "DRILLING"

        elif self.state == "DRILLING":
            if not self.sensors["clamp_closed"]:
                self.report_fault(
                    "Safety interlock: clamp is not confirmed closed."
                )
                self.actuators["drill"] = False
                self.state = "FAULT"
                self.validate_twin()
                return

            self.actuators["drill"] = True

            if not self.validate_safety():
                self.state = "FAULT"
                self.validate_twin()
                return

            self.drill_timer += 1
            self.log(
                f"Drilling progress: {self.drill_timer}/{self.drill_delay}"
            )

            if self.drill_timer > self.max_drill_time:
                self.report_fault(
                    "Timing violation: drilling exceeded "
                    f"{self.max_drill_time} cycles."
                )

            if self.drill_timer >= self.drill_delay:
                self.actuators["drill"] = False
                self.sensors["drill_complete"] = True
                self.log("Drilling completed.")
                self.state = "UNCLAMPING"

        elif self.state == "UNCLAMPING":
            self.actuators["clamp"] = False
            self.sensors["clamp_closed"] = False
            self.log("Clamp opened.")
            self.state = "RELEASING"

        elif self.state == "RELEASING":
            self.actuators["conveyor"] = True
            self.log("Conveyor releasing finished part.")
            self.state = "COMPLETE"

        elif self.state == "COMPLETE":
            self.actuators["conveyor"] = False
            self.product_count += 1
            self.completed = True
            self.log("Production cycle completed.")
            self.validate_twin()
            return True

        elif self.state == "FAULT":
            self.actuators["conveyor"] = False
            self.actuators["clamp"] = False
            self.actuators["drill"] = False
            self.validate_twin()
            return True

        self.validate_twin()
        return False

    def save_dataset(self):
        if not self.dataset_rows:
            print("No data collected; CSV was not changed.")
            return

        file_exists = os.path.exists(self.csv_file)
        needs_header = (
            not file_exists or os.path.getsize(self.csv_file) == 0
        )

        with open(
            self.csv_file, "a", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(
                file, fieldnames=list(self.dataset_rows[0].keys())
            )

            if needs_header:
                writer.writeheader()

            writer.writerows(self.dataset_rows)

        print(
            f"Saved {len(self.dataset_rows)} rows to "
            f"{os.path.abspath(self.csv_file)}"
        )

    def show_report(self):
        self.save_dataset()

        print("\n========== DIGITAL TWIN REPORT ==========")
        print(f"Test mode           : {self.mode}")
        print(f"Final state         : {self.state}")
        print(f"Products completed  : {self.product_count}")
        print(f"Faults detected     : {len(self.faults)}")
        print(f"Sensor mismatches   : {self.mismatch_count}")

        if self.faults:
            for number, fault in enumerate(self.faults, start=1):
                print(f"{number}. {fault}")
            print("Result: FAULT DETECTED")
        elif self.completed:
            print("Result: PASS - no rule faults detected.")
        else:
            print("Result: INCOMPLETE")

        print("==========================================")

    def run(self):
        self.log("Starting factory simulation and Digital Twin.")

        for _ in range(20):
            finished = self.step()
            time.sleep(0.5)

            if finished:
                break

        self.show_report()


def main():
    print("\nDIGITAL TWIN - PHASE 4")
    print("1. Normal operation")
    print("2. Slow drilling test")
    print("3. Part sensor failure test")
    print("4. Clamp safety interlock test")

    choice = input("Select mode (1-4): ").strip()

    if choice not in ("1", "2", "3", "4"):
        print("Invalid choice. Select 1, 2, 3 or 4.")
        return

    factory = FactorySimulation(int(choice))
    factory.run()


if __name__ == "__main__":
    main()
