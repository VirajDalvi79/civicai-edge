# CivicOps Edge: Raspberry Pi 4 setup, from a blank SD card to live events

Do the phases in order. Each one ends with a **✅ check**. Don't move on until it passes; it's much easier to debug one layer at a time.

| Phase | What | Time |
|---|---|---|
| A | Flash the OS (on your PC) | 15 min |
| B | Connect PC ↔ Pi over SSH | 10 min |
| C | Update + enable camera / UART / I2C | 10 min |
| D | Get the code onto the Pi + install | 20–40 min |
| E | Wire + test each sensor, one at a time | 30 min |
| F | Model, calibration, first real run | 15 min |
| G | Send events to your PC + autostart | 10 min |

**Things you need:** Pi 4, a microSD card (16 GB+, A1/A2 class), an official 5.1 V 3 A USB-C supply (a phone charger causes under-voltage throttling that kills inference speed), a heatsink or fan, and a microSD card reader for the PC.

---

## Phase A: Flash the OS (on your PC)

1. Install **Raspberry Pi Imager** from raspberrypi.com/software.
2. Choose: **Device** Raspberry Pi 4 → **OS** *Raspberry Pi OS (other)* → **Raspberry Pi OS Lite (64-bit)** → **Storage** your SD card.
   - Use *Lite*: no desktop, more RAM and CPU for YOLO. It must be **64-bit** because ultralytics/torch need aarch64.
3. When it asks about **customisation**, fill in *everything*. This is what makes it work headless:
   - Hostname: `civicops-pi`
   - Username / password: `civicai` / (your choice). The systemd file assumes `civicai`.
   - Wi-Fi: SSID + password. **Use your phone's hotspot** for now (see note below). Wireless LAN country: `IN`.
   - Locale: `Asia/Kolkata`, keyboard `us`.
   - **Services → Enable SSH → Use password authentication**.
4. Write, eject, put the card in the Pi, power it on. The first boot takes ~2 min (it resizes the filesystem and reboots once).

> **Why a phone hotspot?** College and office Wi-Fi usually blocks devices from talking to each other ("client isolation"), and `.local` names don't resolve there. A hotspot with both the PC and the Pi connected is the most reliable way to start. You can switch to other Wi-Fi later with `sudo nmtui`.

✅ **Check:** your hotspot's "connected devices" list shows `civicops-pi`.

---

## Phase B: Connect from your PC

### Option 1: Same Wi-Fi / hotspot (recommended)
Put the PC on the same hotspot, open **PowerShell**:
```powershell
ping civicops-pi.local
ssh civicai@civicops-pi.local
```
If `.local` doesn't resolve, use the IP shown in the hotspot's device list: `ssh vijay@192.168.x.x`.

### Option 2: Direct Ethernet cable (no Wi-Fi at all)
Plug a LAN cable from the PC to the Pi. Both sides fall back to link-local addresses (169.254.x.x), and `ssh civicai@civicops-pi.local` usually works after ~1 min. If it doesn't: on Windows, share your internet via *Network Connections → Wi-Fi → Properties → Sharing → allow other users…* and pick the Ethernet adapter. The Pi then gets a `192.168.137.x` address.

### Option 3: Monitor + keyboard
Plug in a micro-HDMI monitor + USB keyboard, log in, and run `hostname -I` to get the IP for SSH later.

### Make it comfortable (do this once)
**Passwordless SSH key.** Run in PowerShell on the PC:
```powershell
ssh-keygen -t ed25519            # press Enter for all prompts
type $env:USERPROFILE\.ssh\id_ed25519.pub | ssh civicai@civicops-pi.local "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys"
```
**VS Code Remote-SSH:** install the *Remote - SSH* extension → `F1` → *Remote-SSH: Connect to Host* → `civicai@civicops-pi.local`. You then edit files on the Pi directly with a terminal inside VS Code. This is the best dev loop for this project.

✅ **Check:** `ssh civicai@civicops-pi.local` logs in without asking for a password.

---

## Phase C: Update + enable interfaces (on the Pi)

```bash
sudo apt update && sudo apt full-upgrade -y

# I2C on (for the MPU-6050 later), serial HARDWARE on, serial LOGIN CONSOLE off (GPS needs the port)
sudo raspi-config nonint do_i2c 0
sudo raspi-config nonint do_serial_hw 0
sudo raspi-config nonint do_serial_cons 1

# Give the GPS the good UART (PL011) instead of the clock-dependent mini-UART
echo "dtoverlay=disable-bt" | sudo tee -a /boot/firmware/config.txt
sudo systemctl disable --now hciuart 2>/dev/null

sudo apt install -y python3-picamera2 python3-lgpio python3-venv python3-dev \
                    git i2c-tools rpicam-apps-lite
sudo usermod -aG gpio,i2c,dialout,video $USER
sudo reboot
```
> `disable-bt` turns Bluetooth off. You don't need it here, and it makes `/dev/serial0` the stable UART.

✅ **Check** after the reboot:
```bash
ls -l /dev/serial0          # -> ttyAMA0
ls /dev/i2c-1               # exists
```

---

## Phase D: Code onto the Pi + install

**Getting the code there.** Pick one:
- **Git (best):** push `civicops-edge/` to a GitHub repo from your PC, then on the Pi: `git clone <repo-url> ~/civicops-edge`
- **scp from the PC (PowerShell):** `scp -r C:\Projects\civicai\civicops-edge civicai@civicops-pi.local:~/`

**Install** (on the Pi):
```bash
cd ~/civicops-edge
python3 -m venv --system-site-packages .venv     # --system-site-packages = sees apt's picamera2 + lgpio
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt                  # torch + ultralytics: 10-20 min on a Pi 4
```
> On a **2 GB Pi 4**, if pip gets "Killed", add swap first: `sudo dphys-swapfile swapoff; sudo sed -i 's/CONF_SWAPSIZE=.*/CONF_SWAPSIZE=2048/' /etc/dphys-swapfile; sudo dphys-swapfile setup; sudo dphys-swapfile swapon` (on newer images use `sudo apt install -y systemd-zram-generator` instead if dphys-swapfile is absent).

✅ **Check:** the full test suite passes on the Pi with no hardware attached:
```bash
MOCK_HARDWARE=1 python -m pytest -q tests/
python main.py --mock --duration 20              # watch simulated events
```

---

## Phase E: Wiring (power OFF for every change)

### Pin map (Pi 4, physical pin numbers)
```
            3.3V [ 1] [ 2] 5V      <- HC-SR04 VCC
 (IMU) SDA GPIO2 [ 3] [ 4] 5V      <- GPS VCC
 (IMU) SCL GPIO3 [ 5] [ 6] GND     <- HC-SR04 GND
                 [ 7] [ 8] GPIO14 TXD -> GPS RX (optional)
      (IMU) GND  [ 9] [10] GPIO15 RXD <- GPS TX
                 [11] [12]
                 [13] [14] GND     <- GPS GND
                 [15] [16] GPIO23  -> HC-SR04 TRIG
                 [17] [18] GPIO24  <- HC-SR04 ECHO (via divider!)
                 [19] [20] GND     <- divider 2kΩ bottom
```

### 1. Camera (Pi Camera v2 NoIR), do this first
- Power off. Gently lift the black latch on the **CAMERA** connector (between the micro-HDMI ports and the audio jack).
- Insert the ribbon with the **silver contacts facing the HDMI ports** and the blue strip facing the USB/Ethernet side. Push the latch down.
```bash
rpicam-hello --list-cameras        # must list imx219
python hw_check.py camera          # saves output/check_raw.jpg + check_clahe_gray.jpg
```
Copy the snapshots to your PC to look at them: `scp civicai@civicops-pi.local:~/civicops-edge/output/check_*.jpg .`

### 2. GPS (NEO-6M, GY-NEO6MV2 board)
| GPS pin | Pi pin |
|---|---|
| VCC | Pin 4 (5V). The board has a 3.3V regulator; for a bare module without one use Pin 1 (3.3V) |
| GND | Pin 14 |
| TX | **Pin 10** (GPIO15 / RXD). GPS TX goes to Pi RX |
| RX | Pin 8 (GPIO14 / TXD), optional |

```bash
sudo cat /dev/serial0              # should scroll $GPRMC / $GPGGA lines (Ctrl+C)
python hw_check.py gps
```
A first fix needs **open sky**. Indoors you'll see sentences but no fix. A cold start can take 1–10 min, and the blue LED on the board blinks once it has a fix. Until then the daemon uses the Mira Road/WEH synthetic route and marks `is_mock: true`.

### 3. HC-SR04. ⚠️ Not until you have resistors
**ECHO outputs 5 V. Pi GPIO is 3.3 V max.** A direct connection can permanently damage GPIO24 or the Pi.

```
HC-SR04 ECHO ──[ 1kΩ ]──┬──────────► Pin 18 (GPIO24)   (≈3.3 V)
                        │
                     [ 2kΩ ]        (2kΩ = two 1kΩ in series is fine)
                        │
                       GND (Pin 20)
```
| HC-SR04 | Pi |
|---|---|
| VCC | Pin 2 (5V) |
| TRIG | Pin 16 (GPIO23), direct is fine (Pi → sensor) |
| ECHO | through the divider above → Pin 18 |
| GND | Pin 6 |

Other safe resistor pairs: 10k/20k, 4.7k/10k. Or buy an **HC-SR04P / RCWL-1601**: power it from **Pin 1 (3.3 V)** and it needs no divider.

Mount it **pointing straight down** at the road, 20–40 cm above it (the HC-SR04's minimum range is 2 cm, and its beam is ~15° wide).
```bash
python hw_check.py ultrasonic      # hold a book under it and move it
```

### 4. MPU-6050 (when you get one)
VCC → Pin 1 (3.3 V), GND → Pin 9, SDA → Pin 3, SCL → Pin 5. Mount it with the chip's **Z axis vertical**.
```bash
i2cdetect -y 1                     # "68" should appear
python hw_check.py imu             # ~1.00 g at rest; tap the board to see spikes
```
Without it, the daemon logs `IMU DISABLED`, sends `z_impact_g: null`, and runs normally.

✅ **Check:** `python hw_check.py all` → `camera=OK ultrasonic=OK imu=OK gps=OK` (imu reports OK/SKIP when it isn't fitted).

---

## Phase F: Model, calibration, first real run

```bash
source .venv/bin/activate
python -m model.download_assets --pothole     # pothole-trained YOLOv8n -> model/pothole_ncnn_model/
python -m model.download_assets --benchmark   # expect roughly 80-150 ms per frame at 320px on a Pi 4
python hw_check.py detector                   # point the camera at a pothole photo on your phone
```
> Stock `yolov8n` (COCO, 80 everyday classes) **has no pothole class**. `--pothole` fetches a community YOLOv8n model trained on potholes (Hugging Face `keremberke/yolov8n-pothole-segmentation`). For the best accuracy on Mumbai roads, fine-tune your own model later on a public pothole dataset plus your own NoIR captures (Colab, ~1 h), then run `python -m model.download_assets --weights best.pt`.

**Calibrate the ultrasonic baseline:** park on flat road, sensor mounted at its final height:
```bash
python main.py --calibrate                    # saves calibration/ultrasonic_baseline.json
```

**First run:**
```bash
python main.py                                # Ctrl+C = graceful shutdown
tail -f output/events.jsonl                   # in a second SSH window
```
Every 10 s it logs FPS, inference time, the backend for each sensor, and event counts. Annotated snapshots of each event go to `output/event_images/`.

---

## Phase G: Events on your PC + autostart

**Watch events live on the PC:**
```powershell
# On the PC (Python installed):
python desktop_receiver.py          # allow it through Windows Firewall (Private)
ipconfig                            # note the PC's IPv4 on the hotspot, e.g. 192.168.43.20
```
```bash
# On the Pi:
TELEMETRY_URL=http://192.168.43.20:8000/events python main.py
```
Events print in colour on the PC and are saved to `received_events.jsonl`. If the PC is off, the Pi keeps them in `output/unsent.jsonl` and re-sends them automatically when the connection returns.

**Start on boot:**
```bash
sudo cp deploy/civicops-edge.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now civicops-edge
journalctl -u civicops-edge -f
```

---

## The NoIR pink/purple tint: what to do

The v2 **NoIR** has no infrared-blocking filter. In daylight, IR light leaks into the red/blue channels, so everything looks pink or magenta. Fixes, from cheapest to cleanest:

| Fix | How | Colour? | Notes |
|---|---|---|---|
| **1. NoIR tuning file** (on by default) | `CAMERA_NOIR_TUNING=1` loads libcamera's `imx219_noir.json` | Much better | Free; tuned by Raspberry Pi specifically for this sensor |
| **2. Grayscale + CLAHE for YOLO** (default) | `PREPROCESS_MODE=clahe_gray` | Removed entirely | Potholes are shape/shadow/texture, so colour barely matters. Tested here: a stock YOLOv8n lost only ~0.03 confidence on grayscale. Day and IR-night frames look alike to the model |
| 3. Gray-world white balance | `PREPROCESS_MODE=grayworld` | Mostly neutral | Keeps colour; weaker under mixed lighting |
| 4. Hardware IR-cut filter | 650 nm IR-cut lens filter in front of the lens (~₹200–400) | True colour | Cleanest daytime image, but you lose night-with-IR |
| 5. Swap camera | Camera Module v2/v3 (non-NoIR) | True colour | Only if you'll never run at night |

**Recommendation:** keep the NoIR with **1 + 2** (the code's defaults). If you add an **850 nm IR illuminator** later, the same grayscale pipeline works at night with no changes. That's the real reason to have a NoIR. If you fine-tune your own model, train it on grayscale images too so training matches deployment.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ssh: Could not resolve hostname civicops-pi.local` | Use the IP from the hotspot list; make sure the PC is on the same network |
| `rpicam-hello` finds no cameras | Ribbon reversed or not fully seated. Power off, reseat (contacts toward HDMI) |
| `numpy.dtype size changed` / picamera2 import error in the venv | `pip uninstall -y numpy` inside the venv so apt's numpy (matching picamera2) is used |
| GPS: `sudo cat /dev/serial0` shows nothing | TX/RX swapped, or serial console still enabled (re-run Phase C), or baud isn't 9600 |
| GPS: sentences but `fix=False` | Needs open sky. Wait up to 10 min on a cold start |
| Ultrasonic always `None` / many timeouts | Check the divider and TRIG/ECHO pins; the target must be 2–400 cm away and hard-surfaced |
| `i2cdetect` shows no 68 | SDA/SCL swapped, or I2C not enabled (Phase C) |
| Low FPS / `throttled` | `vcgencmd get_throttled` should be `0x0`. Otherwise it's the power supply or heat: use the official PSU plus a heatsink/fan |
| Events have `depth_cm: 0.0` while moving | Tune `FUSION_DELAY_S` (0.3–0.6). The camera sees the pothole before the sensor passes over it |

Every setting in `config.py` can be overridden per run: `FUSION_DELAY_S=0.5 DETECTION_CONFIDENCE=0.5 python main.py`.
