"""Control hands using POS_SET (1474) for thumb bend to match left/right behavior."""
from pymodbus.client import ModbusTcpClient
import time

LEFT_IP = "192.168.123.210"
RIGHT_IP = "192.168.123.211"
PORT = 6000

left_client = ModbusTcpClient(LEFT_IP, port=PORT)
right_client = ModbusTcpClient(RIGHT_IP, port=PORT)

left_client.connect()
right_client.connect()

left_client.write_register(1004, 1)
right_client.write_register(1004, 1)

# Use ANGLE_SET (1486) for all fingers except thumb bend
# Use POS_SET (1474) for thumb bend (index 4) to match both hands
#
# From diagnostic:
#   Left  thumb_bend POS range: ~393 (open) to ~1092 (closed)
#   Right thumb_bend POS range: ~476 (open) to ~1338 (closed)
#
# We use the LEFT hand's range as reference and map it to the right hand.
# POS_SET uses -1 to mean "don't control this joint"

# --- Config ---
LEFT_THUMB_OPEN = 393
LEFT_THUMB_CLOSED = 1092

RIGHT_THUMB_OPEN = 476
RIGHT_THUMB_CLOSED = 1338

# Set desired thumb bend as 0.0 (closed) to 1.0 (open)
thumb_bend_fraction = 0.0  # <-- change this to test (0.0 = closed, 1.0 = open)

left_thumb_pos = int(LEFT_THUMB_OPEN + thumb_bend_fraction * (LEFT_THUMB_CLOSED - LEFT_THUMB_OPEN))
# Map to right hand range so both reach the same physical position
right_thumb_pos = int(RIGHT_THUMB_OPEN + thumb_bend_fraction * (RIGHT_THUMB_CLOSED - RIGHT_THUMB_OPEN))

# Wait, the mapping is inverted: higher POS = more closed
# fraction 0.0 (closed) -> use CLOSED pos, fraction 1.0 (open) -> use OPEN pos
# Right hand needs higher POS to match left physically (mirrored mechanism)
left_thumb_pos = 1092
right_thumb_pos = 1800

print(f"Thumb bend fraction: {thumb_bend_fraction}")
print(f"Left  thumb POS: {left_thumb_pos}")
print(f"Right thumb POS: {right_thumb_pos}")

# Set other fingers via ANGLE_SET (1486)
NO_ACTION = 65535  # -1 as unsigned 16-bit = 0xFFFF

angle_regs = [
    0,         # pinky (closed)
    0,         # ring (closed)
    0,         # middle (closed)
    0,         # index (closed)
    NO_ACTION, # thumb bend: skip (we use POS_SET instead)
    0,         # thumb rotation (closed)
]

left_client.write_registers(1486, angle_regs)
right_client.write_registers(1486, angle_regs)

# Set thumb bend via POS_SET (1474), -1 for all other joints
left_pos_regs = [NO_ACTION, NO_ACTION, NO_ACTION, NO_ACTION, left_thumb_pos, NO_ACTION]
right_pos_regs = [NO_ACTION, NO_ACTION, NO_ACTION, NO_ACTION, right_thumb_pos, NO_ACTION]

left_client.write_registers(1474, left_pos_regs)
right_client.write_registers(1474, right_pos_regs)

left_client.close()
right_client.close()
