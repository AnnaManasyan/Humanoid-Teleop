"""Diagnostic script to compare left and right hand behavior."""
from pymodbus.client import ModbusTcpClient
import time
import struct

RIGHT_IP = "192.168.123.210"
LEFT_IP = "192.168.123.211"
PORT = 6000

left_client = ModbusTcpClient(LEFT_IP, port=PORT)
right_client = ModbusTcpClient(RIGHT_IP, port=PORT)

left_client.connect()
right_client.connect()

# Reset errors
left_client.write_register(1004, 1)
right_client.write_register(1004, 1)

def read_signed(client, addr, cnt):
    """Read registers as signed 16-bit integers."""
    resp = client.read_holding_registers(addr, count=cnt)
    if resp.isError():
        return None
    packed = struct.pack('>' + 'H' * cnt, *resp.registers)
    return list(struct.unpack('>' + 'h' * cnt, packed))

def read_unsigned(client, addr, cnt):
    resp = client.read_holding_registers(addr, count=cnt)
    if resp.isError():
        return None
    return list(resp.registers)

# Read default speed and force settings
print("=== DEFAULT SPEED (1032) ===")
print(f"  Left:  {read_unsigned(left_client, 1032, 6)}")
print(f"  Right: {read_unsigned(right_client, 1032, 6)}")
print()
print("=== DEFAULT FORCE (1044) ===")
print(f"  Left:  {read_unsigned(left_client, 1044, 6)}")
print(f"  Right: {read_unsigned(right_client, 1044, 6)}")
print()
print("=== HAND ID (1000) ===")
print(f"  Left:  {read_unsigned(left_client, 1000, 1)}")
print(f"  Right: {read_unsigned(right_client, 1000, 1)}")
print()

labels = ["pinky", "ring", "middle", "index", "thumb_bend", "thumb_rot"]

# First, open all fingers fully
print("=== Sending OPEN command (1000 for all) ===")
open_cmd = [1000, 1000, 1000, 1000, 800, 1000]
left_client.write_registers(1486, open_cmd)
right_client.write_registers(1486, open_cmd)
time.sleep(3)

print("=== State after OPEN ===")
print(f"  Left  ANGLE_ACT (1546): {read_unsigned(left_client, 1546, 6)}")
print(f"  Right ANGLE_ACT (1546): {read_unsigned(right_client, 1546, 6)}")
print(f"  Left  POS_ACT   (1534): {read_unsigned(left_client, 1534, 6)}")
print(f"  Right POS_ACT   (1534): {read_unsigned(right_client, 1534, 6)}")
print()

# Now close thumb pinch only
print("=== Sending thumb_bend=0 (closed), others=1000 (open) ===")
cmd = [1000, 1000, 1000, 1000, 0, 1000]
left_client.write_registers(1486, cmd)
right_client.write_registers(1486, cmd)
time.sleep(3)

print("=== State after thumb_bend CLOSE ===")
l_angle = read_unsigned(left_client, 1546, 6)
r_angle = read_unsigned(right_client, 1546, 6)
l_pos = read_unsigned(left_client, 1534, 6)
r_pos = read_unsigned(right_client, 1534, 6)
l_force = read_signed(left_client, 1582, 6)
r_force = read_signed(right_client, 1582, 6)
l_current = read_unsigned(left_client, 1594, 6)
r_current = read_unsigned(right_client, 1594, 6)
l_status = read_unsigned(left_client, 1612, 3)
r_status = read_unsigned(right_client, 1612, 3)
l_err = read_unsigned(left_client, 1606, 3)
r_err = read_unsigned(right_client, 1606, 3)

print(f"  Left  ANGLE_ACT: {l_angle}")
print(f"  Right ANGLE_ACT: {r_angle}")
print()
print(f"  Left  POS_ACT:   {l_pos}")
print(f"  Right POS_ACT:   {r_pos}")
print()
print(f"  Left  FORCE_ACT: {l_force}")
print(f"  Right FORCE_ACT: {r_force}")
print()
print(f"  Left  CURRENT:   {l_current}")
print(f"  Right CURRENT:   {r_current}")
print()
print(f"  Left  STATUS:    {l_status}")
print(f"  Right STATUS:    {r_status}")
print()
print(f"  Left  ERROR:     {l_err}")
print(f"  Right ERROR:     {r_err}")
print()

# Highlight differences for thumb_bend (index 4)
if l_angle and r_angle:
    print("=== THUMB BEND (index 4) COMPARISON ===")
    print(f"  Command sent:    0 (closed)")
    print(f"  Left  angle_act: {l_angle[4]}")
    print(f"  Right angle_act: {r_angle[4]}")
    print(f"  Left  pos_act:   {l_pos[4]}")
    print(f"  Right pos_act:   {r_pos[4]}")
    print(f"  Angle difference: {abs(l_angle[4] - r_angle[4])}")
    print(f"  Pos difference:   {abs(l_pos[4] - r_pos[4])}")

# Re-open to safe position
print("\n=== Re-opening hands ===")
left_client.write_registers(1486, open_cmd)
right_client.write_registers(1486, open_cmd)

left_client.close()
right_client.close()
