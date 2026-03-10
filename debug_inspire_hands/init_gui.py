"""Standalone init/config GUI for Inspire hands, compatible with current pymodbus."""
import sys
from pymodbus.client import ModbusTcpClient
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLineEdit, QLabel

IP = sys.argv[1] if len(sys.argv) > 1 else "192.168.123.211"
PORT = 6000

registers = {
    1000: {"name": "HAND_ID", "description": "Hand ID", "length": 1},
    1002: {"name": "REDU_RATIO", "description": "Baud rate setting", "length": 1},
    1032: {"name": "DEFAULT_SPEED_SET", "description": "Default speed (per DOF)", "length": 6},
    1044: {"name": "DEFAULT_FORCE_SET", "description": "Default force threshold (per DOF)", "length": 6},
    1700: {"name": "ip", "description": "IP address", "length": 2},
}

baud_rates = {0: 115200, 1: 57600, 2: 19200, 3: 921600}
baud_rates_reverse = {v: k for k, v in baud_rates.items()}


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.client = ModbusTcpClient(IP, port=PORT)
        if not self.client.connect():
            print(f"Failed to connect to {IP}:{PORT}")
            sys.exit(1)
        print(f"Connected to {IP}:{PORT}")
        # Find device ID
        for i in range(10):
            resp = self.client.read_holding_registers(1000, count=1, device_id=i)
            if not resp.isError():
                self.dev_id = resp.registers[0]
                print(f"Found device ID: {self.dev_id}")
                break
        else:
            self.dev_id = 1
            print("No device found, using ID=1")
        self.client.close()
        self.client = ModbusTcpClient(IP, port=PORT)
        self.client.connect()
        self.initUI()
        self.read_registers()

    def initUI(self):
        self.setWindowTitle(f'Inspire Hand Config ({IP})')
        layout = QVBoxLayout()

        for text, callback in [
            ("Read settings", self.read_registers),
            ("Write settings", self.save_registers),
            ("Save to Flash", lambda: self.client.write_register(1005, 1, device_id=self.dev_id)),
            ("Factory reset", lambda: self.client.write_register(1006, 1, device_id=self.dev_id)),
            ("Calibrate force sensor", self.force_calibrate),
            ("Clear errors", lambda: self.client.write_register(1004, 1, device_id=self.dev_id)),
        ]:
            btn = QPushButton(text)
            btn.clicked.connect(callback)
            layout.addWidget(btn)

        self.register_inputs = {}
        for address, info in registers.items():
            layout.addWidget(QLabel(info['description']))
            n = info['length'] * 2 if info['name'] == 'ip' else info['length']
            inputs = [QLineEdit() for _ in range(n)]
            for inp in inputs:
                layout.addWidget(inp)
            self.register_inputs[address] = inputs

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        self.show()

    def force_calibrate(self):
        self.client.write_registers(1486, [1000] * 6, device_id=self.dev_id)
        self.client.write_register(1009, 1, device_id=self.dev_id)
        print("Force sensor calibration triggered")

    def read_registers(self):
        for address, info in registers.items():
            if info['name'] == 'ip':
                resp = self.client.read_holding_registers(address, count=2, device_id=self.dev_id)
                if not resp.isError():
                    v = resp.registers
                    bytes_ = [v[0] & 0xFF, (v[0] >> 8) & 0xFF, v[1] & 0xFF, (v[1] >> 8) & 0xFF]
                    for j in range(4):
                        self.register_inputs[address][j].setText(str(bytes_[j]))
                    print(f"IP: {'.'.join(map(str, bytes_))}")
            else:
                resp = self.client.read_holding_registers(address, count=info['length'], device_id=self.dev_id)
                if not resp.isError():
                    vals = resp.registers
                    if info['name'] == 'REDU_RATIO':
                        self.register_inputs[address][0].setText(str(baud_rates.get(vals[0], vals[0])))
                    else:
                        for j in range(info['length']):
                            self.register_inputs[address][j].setText(str(vals[j]))
                    print(f"{info['name']}: {vals}")

    def save_registers(self):
        for address, info in registers.items():
            if info['name'] == 'ip':
                vals = [int(inp.text()) for inp in self.register_inputs[address]]
                shorts = [(vals[1] << 8) | vals[0], (vals[3] << 8) | vals[2]]
                self.client.write_registers(address, shorts, device_id=self.dev_id)
            elif info['length'] == 1:
                val = int(self.register_inputs[address][0].text())
                if info['name'] == 'REDU_RATIO':
                    val = baud_rates_reverse.get(val, val)
                self.client.write_register(address, val, device_id=self.dev_id)
            else:
                vals = [int(inp.text()) for inp in self.register_inputs[address]]
                self.client.write_registers(address, vals, device_id=self.dev_id)
        print("Settings written")

    def closeEvent(self, event):
        self.client.close()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())
