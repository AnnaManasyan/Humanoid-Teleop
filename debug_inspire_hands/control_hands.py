from pymodbus.client import ModbusTcpClient

LEFT_IP = "192.168.123.210"
RIGHT_IP = "192.168.123.211"
PORT = 6000

left_client = ModbusTcpClient(LEFT_IP, port=PORT)
right_client = ModbusTcpClient(RIGHT_IP, port=PORT)

left_client.connect()
right_client.connect()

left_client.write_register(1004, 1)
right_client.write_register(1004, 1)

left_regs = [
    1000,  # pinky
    1000,  # ring
    1000,  # middle
    1000,  # index
    850,  # thumb pinch
    1000,  # thumb rotation
]
# The actual range for the commands is 0-1000.
# 0 is closed, 1000 is fully opened.
# Please do not exceed 850 for the thumb pinch because it will crack.
# Closing is not an issue.

right_regs = [x for x in left_regs]
right_regs[-2]=50

# left_client.write_registers(1486, left_regs)
right_client.write_registers(1486, right_regs)

left_client.close()
right_client.close()