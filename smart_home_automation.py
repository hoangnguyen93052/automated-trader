import datetime
import json
import os
import random
import time
from typing import List, Dict

class Device:
    def __init__(self, name: str, device_type: str, status: bool = False):
        self.name = name
        self.device_type = device_type
        self.status = status

    def toggle(self):
        self.status = not self.status
        print(f"{self.name} is now {'ON' if self.status else 'OFF'}.")

    def __str__(self):
        return f"{self.name} (Type: {self.device_type}, Status: {'ON' if self.status else 'OFF'})"


class SmartHome:
    def __init__(self):
        self.devices: List[Device] = []
        self.schedule: Dict[str, List[str]] = {}
        self.notifications: List[str] = []

    def add_device(self, device: Device):
        self.devices.append(device)
        print(f"Added device: {device}")

    def remove_device(self, device_name: str):
        self.devices = [device for device in self.devices if device.name != device_name]
        print(f"Removed device: {device_name}")

    def toggle_device(self, device_name: str):
        for device in self.devices:
            if device.name == device_name:
                device.toggle()
                return
        print(f"Device {device_name} not found.")

    def schedule_device(self, device_name: str, action: str, time_str: str):
        if device_name not in self.schedule:
            self.schedule[device_name] = []
        self.schedule[device_name].append(f"{datetime.datetime.now().isoformat()}: {action} at {time_str}")

    def process_schedule(self):
        current_time = datetime.datetime.now().isoformat()
        for device_name, actions in list(self.schedule.items()):
            for action in actions:
                action_time = action.split(" at ")[-1]
                if action_time <= current_time:
                    self.toggle_device(device_name)
                    actions.remove(action)
                    self.notifications.append(f"Executed '{action}' for {device_name}.")

    def get_notifications(self):
        return self.notifications

    def save_to_file(self, filename: str):
        with open(filename, 'w') as file:
            json.dump(self.schedule, file)
        print(f"Schedule saved to {filename}")

    def load_from_file(self, filename: str):
        if os.path.exists(filename):
            with open(filename, 'r') as file:
                self.schedule = json.load(file)
            print(f"Schedule loaded from {filename}")
        else:
            print(f"File {filename} does not exist.")

    def __str__(self):
        return "\n".join(str(device) for device in self.devices)


def main():
    home = SmartHome()

    home.add_device(Device("Living Room Light", "Light"))
    home.add_device(Device("Thermostat", "Temperature Control"))
    home.add_device(Device("Security Camera", "Camera"))

    print("\nCurrent Devices:")
    print(home)

    home.toggle_device("Living Room Light")
    home.schedule_device("Living Room Light", "Turn ON", (datetime.datetime.now() + datetime.timedelta(seconds=5)).isoformat())

    print("\nProcessing schedule:")
    time.sleep(6)
    home.process_schedule()

    print("\nNotifications:")
    print(home.get_notifications())

    home.save_to_file("schedule.json")
    home.load_from_file("schedule.json")


if __name__ == "__main__":
    main()