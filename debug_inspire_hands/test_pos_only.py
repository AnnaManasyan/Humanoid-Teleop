"""Test POS_SET limits for all non-thumb fingers on both hands."""
from pymodbus.client import ModbusTcpClient
import time

LEFT_IP = "192.168.123.210"
RIGHT_IP = "192.168.123.211"
PORT = 6000

left = ModbusTcpClient(LEFT_IP, port=PORT)
right = ModbusTcpClient(RIGHT_IP, port=PORT)
left.connect()
right.connect()
left.write_register(1004, 1)
right.write_register(1004, 1)

NO = 65535
labels = ["pinky", "ring", "middle", "index"]

def read_pos(client):
    resp = client.read_holding_registers(1534, count=6)
    return list(resp.registers) if not resp.isError() else None

# Step 1: Open fingers 0-3 via ANGLE_SET, leave thumbs (4,5) untouched
print("Step 1: Opening fingers 0-3 via ANGLE_SET (800), thumbs unchanged (-1)")
left.write_registers(1486, [800, 800, 800, 800, NO, NO])
right.write_registers(1486, [800, 800, 800, 800, NO, NO])
time.sleep(2)
print(f"  Left  POS_ACT: {read_pos(left)}")
print(f"  Right POS_ACT: {read_pos(right)}")

# Step 2: Close fingers 0-3 via POS_SET to a high value, thumbs unchanged
print("\nStep 2: Closing fingers 0-3 via POS_SET (1800), thumbs unchanged")
left.write_registers(1474, [1800, 1800, 1800, 1800, NO, NO])
right.write_registers(1474, [1800, 1800, 1800, 1800, NO, NO])
time.sleep(2)
l_pos = read_pos(left)
r_pos = read_pos(right)
print(f"  Left  POS_ACT: {l_pos}")
print(f"  Right POS_ACT: {r_pos}")

print("\n=== MECHANICAL LIMITS (fingers 0-3) ===")
for i in range(4):
    print(f"  {labels[i]:8s}  Left: {l_pos[i]:4d}  Right: {r_pos[i]:4d}  Diff: {abs(l_pos[i] - r_pos[i])}")

left.close()
right.close()
