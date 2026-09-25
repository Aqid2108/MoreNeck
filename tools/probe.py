"""Read-only probe of the servo bus: PING + READ only, never writes.

Usage: .venv/bin/python tools/probe.py
"""
import time

import serial
from serial.tools import list_ports

VID, PID = 0x1A86, 0x55D3


def find_port():
    for p in list_ports.comports():
        if p.vid == VID and p.pid == PID:
            return p.device
    raise SystemExit(f"No USB serial device {VID:04x}:{PID:04x}. Is the bus adapter plugged in?")


def packet(sid, instr, params=b""):
    body = bytes([sid, len(params) + 2, instr]) + params
    return b"\xff\xff" + body + bytes([(~sum(body)) & 0xFF])


def txrx(ser, pkt, timeout=0.03):
    ser.reset_input_buffer()
    ser.write(pkt)
    ser.flush()
    deadline = time.time() + timeout
    buf = b""
    while time.time() < deadline:
        buf += ser.read(ser.in_waiting or 1)
        i = buf.find(b"\xff\xff")
        if i >= 0 and len(buf) >= i + 4:
            n = buf[i + 3]
            if len(buf) >= i + 4 + n:
                r = buf[i:i + 4 + n]
                if r[2] == pkt[2]:
                    return r
                buf = buf[i + 4 + n:]
    return None


def read(ser, sid, addr, n):
    r = txrx(ser, packet(sid, 0x02, bytes([addr, n])))
    if not r:
        return None, None
    return r[4], r[5:5 + n]


def u16(b, i):
    return b[i] | (b[i + 1] << 8)


def signed15(v):
    return -(v & 0x7FFF) if v & 0x8000 else v


def main():
    port = find_port()
    for baud in (1_000_000, 500_000, 115_200):
        with serial.Serial(port, baud, timeout=0.01) as ser:
            found = [sid for sid in range(0, 21) if txrx(ser, packet(sid, 0x01))]
            print(f"{port} @ {baud}: responding IDs {found}")
            if not found:
                continue
            for sid in found:
                _, head = read(ser, sid, 0x00, 7)
                _, ee = read(ser, sid, 0x09, 28)
                err, rt = read(ser, sid, 0x38, 15)
                _, te = read(ser, sid, 0x28, 1)
                _, tl = read(ser, sid, 0x30, 2)
                print(f"\n== ID {sid} (packet error {err}) ==")
                if head:
                    print(f"  firmware {head[0]}.{head[1]}  model {u16(head, 3):#06x}")
                if ee:
                    print(f"  angle limits : {u16(ee, 0)} .. {u16(ee, 2)}")
                    print(f"  max temp     : {ee[4]} C")
                    print(f"  volt limits  : {ee[6] / 10:.1f} .. {ee[5] / 10:.1f} V")
                    print(f"  max torque   : {u16(ee, 7)}")
                    print(f"  prot current : {u16(ee, 0x1C - 9) * 6.5:.0f} mA")
                    print(f"  mode         : {ee[0x21 - 9]}")
                if rt:
                    print(f"  position     : {u16(rt, 0)}")
                    print(f"  speed        : {signed15(u16(rt, 2))}")
                    print(f"  load         : {u16(rt, 4)}")
                    print(f"  voltage      : {rt[6] / 10:.1f} V")
                    print(f"  temperature  : {rt[7]} C")
                    print(f"  status byte  : {rt[9]:#04x}")
                    print(f"  moving       : {rt[10]}")
                    print(f"  current      : {abs(signed15(u16(rt, 13))) * 6.5:.0f} mA")
                if te:
                    print(f"  torque enable: {te[0]}")
                if tl:
                    print(f"  torque limit : {u16(tl, 0)} (RAM reg 48)")
            return


if __name__ == "__main__":
    main()
