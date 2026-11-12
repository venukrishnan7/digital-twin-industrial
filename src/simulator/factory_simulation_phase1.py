
import time
from datetime import datetime

class FactorySimulation:
    def __init__(self,drill_delay=2,sensor_failure=False):
        self.state="IDLE"
        self.time_step=0
        self.drill_delay=drill_delay
        self.sensor_failure=sensor_failure
        self.drill_timer=0
        self.product_count=0
        self.sensors={
            "part_detected":False,
            "clamp_closed":False,
            "drill_complete":False
        }
        self.actuators={
            "conveyor":False,
            "clamp":False,
            "drill":False
        }

    def log(self,message):
        timestamp=datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}")

    def show_status(self):
        print("\n--- FACTORY STATUS ---")
        print(f"Step       : {self.state}")
        print(f"Products   : {self.product_count}")
        print(f"Sensors    : {self.sensors}")
        print(f"Actuators  : {self.actuators}")
        print("----------------------\n")

    def step(self):
        self.time_step+=1
        self.log(f"Cycle {self.time_step}: {self.state}")

        if self.state=="IDLE":
            self.actuators["conveyor"]=True
            self.log("Conveyor started.")
            self.state="CONVEYING"

        elif self.state=="CONVEYING":
            self.sensors["part_detected"]=not self.sensor_failure

            if self.sensors["part_detected"]:
                self.actuators["conveyor"]=False
                self.log("Part detected. Conveyor stopped.")
                self.state="CLAMPING"
            else:
                self.log("WARNING: Part detection sensor failed.")
                self.log("Waiting for part detection.")

        elif self.state=="CLAMPING":
            self.actuators["clamp"]=True
            self.sensors["clamp_closed"]=True
            self.log("Clamp closed. Part secured.")
            self.state="DRILLING"

        elif self.state=="DRILLING":
            self.actuators["drill"]=True
            self.drill_timer+=1
            self.log(
                f"Drilling in progress: "
                f"{self.drill_timer}/{self.drill_delay}"
            )

            if self.drill_timer>=self.drill_delay:
                self.actuators["drill"]=False
                self.sensors["drill_complete"]=True
                self.log("Drilling completed.")
                self.state="UNCLAMPING"

        elif self.state=="UNCLAMPING":
            self.actuators["clamp"]=False
            self.sensors["clamp_closed"]=False
            self.log("Clamp opened.")
            self.state="RELEASING"

        elif self.state=="RELEASING":
            self.actuators["conveyor"]=True
            self.log("Conveyor moving finished part.")
            self.state="COMPLETE"

        elif self.state=="COMPLETE":
            self.actuators["conveyor"]=False
            self.product_count+=1
            self.log(
                f"Production cycle complete. "
                f"Total products: {self.product_count}"
            )
            self.state="IDLE"
            self.sensors["part_detected"]=False
            self.sensors["drill_complete"]=False
            self.drill_timer=0
            return True

        return False

    def run(self):
        self.log("Starting Digital Twin factory simulation.")
        self.log("Press Ctrl+C to stop the simulation.")

        try:
            while True:
                completed=self.step()
                self.show_status()

                if completed:
                    time.sleep(2)
                else:
                    time.sleep(1)

        except KeyboardInterrupt:
            self.log("Simulation stopped by user.")
            self.show_status()


def main():
    print("\nDIGITAL TWIN - FACTORY SIMULATION")
    print("1. Normal operation")
    print("2. Slow drilling operation")
    print("3. Part detection sensor failure")

    choice=input("Select a mode (1/2/3): ").strip()

    if choice=="1":
        factory=FactorySimulation()
    elif choice=="2":
        factory=FactorySimulation(drill_delay=5)
    elif choice=="3":
        factory=FactorySimulation(sensor_failure=True)
    else:
        print("Invalid choice. Please select 1, 2 or 3.")
        return

    factory.run()


if __name__=="__main__":
    main()
