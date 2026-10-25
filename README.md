# Offline Handwritten Shape Recognition — PC to Raspberry Pi Pico

Draw a shape with your mouse on the PC, a small CNN recognizes it, and the shape **name** is sent over USB serial to a Raspberry Pi Pico, which shows it on an SSD1306 OLED display.

Everything runs **fully offline**. No internet and no downloaded dataset are needed.

```
 ┌──────────────────────────┐   USB serial    ┌───────────────────────┐
 │ PC: shape_cnn_app.py     │  "pentagon\n"   │ Pico: main.py         │
 │  Tkinter canvas → CNN →  │ ──────────────► │  reads line → shows   │
 │  predicted shape name    │                 │  name on SSD1306 OLED │
 └──────────────────────────┘                 └───────────────────────┘
```

## Features

- Recognizes **19 shapes**: circle, oval, triangle, square, rectangle, parallelogram, rhombus, trapezoid, kite, pentagon, hexagon, heptagon, octagon, nonagon, decagon, star, heart, crescent, semicircle
- **Synthetic training data**: the script draws thousands of randomized outline shapes (random size, rotation, stroke width) instead of using a downloaded dataset
- **Auto-train on first run**, saved to `shape_cnn.pth`, and loaded instantly on every later run
- Tkinter GUI with a drawing canvas, **Predict** and **Clear** buttons, and the **top-3 predictions** with confidence
- COM port selection, refresh and connect inside the GUI
- Uses the GPU automatically if CUDA is available, otherwise the CPU

## Project files

| File | Runs on | Purpose |
|------|---------|---------|
| `shape_cnn_app.py` | PC | Dataset generator, CNN, training, prediction, GUI and serial sender (single file) |
| `pico_shape_main.py` | Pico | Save on the Pico as `main.py`. Reads shape names from USB serial and shows them on the OLED |
| `requirements_shapes.txt` | PC | Python dependencies |
| `shape_cnn.pth` | PC | Trained model (auto-created on first run) |
| `ssd1306.py` | Pico | Official MicroPython SSD1306 driver (**you must add this yourself**) |

## Hardware

- Raspberry Pi Pico (MicroPython installed)
- SSD1306 OLED, 128×64, I2C
- USB cable (data-capable, not charge-only)

### Wiring

| SSD1306 | Pico |
|---------|------|
| SDA | GPIO 4 |
| SCL | GPIO 5 |
| VCC | 3V3 |
| GND | GND |

## PC setup

Requires Python 3.8+ with Tkinter (included with the standard Python installer on Windows and macOS).

```bash
pip install -r requirements_shapes.txt
```

Dependencies: `torch`, `Pillow`, `numpy`, `pyserial`.

## Pico setup

1. Install MicroPython firmware on the Pico.
2. Copy the official `ssd1306.py` driver to the Pico.
3. Copy `pico_shape_main.py` to the Pico and **rename it to `main.py`** so it runs automatically on boot.
4. Reset the Pico. The OLED should show **"Shape Receiver / Waiting for PC..."**.

## Usage

```bash
python shape_cnn_app.py
```

1. **First run only:** the app generates the synthetic dataset and trains the CNN (12 epochs, 900 training and 120 validation samples per class). This can take a few minutes on CPU. Progress and accuracy are printed in the terminal.
2. In **Pico Serial Connection**, choose the Pico's COM port (press **Refresh** if it isn't listed) and click **Connect**.
3. Draw a shape on the white canvas.
4. Click **PREDICT**. The prediction and the top-3 list appear in the window, and the winning shape name is sent to the Pico automatically.
5. Click **CLEAR** to draw another shape.

If the Pico isn't connected, the prediction still works and the app shows a "Not sent" warning.

## How it works

### 1. Synthetic data
Each shape is first rasterized as a filled mask with randomized size, rotation and proportions. It is then converted to an **outline** (using repeated erosion and subtraction), with a random stroke width of 2–4 px, because a pen stroke on the canvas is an outline, not a filled region.

### 2. CNN model (`ShapeCNN`)
- Input: 64×64 grayscale image
- 3 × (Conv 3×3 → ReLU → MaxPool): 32, 64, 128 channels
- Dropout 0.25 → Flatten → FC 256 → Dropout 0.5 → FC 19
- Adam optimizer (lr 1e-3), cross-entropy loss, batch size 128

### 3. Preprocessing
The canvas image (black strokes on white) is inverted to white on black, cropped to the drawn region, scaled to fit with the aspect ratio kept, and centered on a 64×64 frame. This matches the training data.

### 4. Serial protocol
The PC sends one lowercase shape name per line, UTF-8, ending with `\n`, for example `pentagon\n`. The Pico accepts only the 19 valid names. Anything else shows "Invalid data:" on the OLED. Input longer than 32 characters is discarded.

## Configuration

Constants at the top of `shape_cnn_app.py`:

| Constant | Default | Meaning |
|----------|---------|---------|
| `IMG_SIZE` | 64 | Model input resolution |
| `CANVAS_SIZE` | 320 | On-screen drawing canvas size |
| `BRUSH_RADIUS` | 8 | Pen thickness |
| `BAUD_RATE` | 9600 | Serial baud rate (the Pico's USB serial ignores the value, but it is set for completeness) |

Training settings are passed in `main()`: `train_and_save_model(epochs=12, samples_per_class=900, val_per_class=120)`.

## Retraining

Delete `shape_cnn.pth` and run the app again. You **must** retrain if you change `SHAPE_CLASSES`. The class order is stored in the checkpoint, and the app raises an error on a mismatch.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| No COM port listed | Use a data USB cable, check that the Pico is detected, then press **Refresh** |
| "Could not connect" error | Close Thonny or any other program using the Pico's port. Only one program can open it at a time |
| OLED stays blank | Check SDA/SCL wiring (GPIO 4/5), that `ssd1306.py` is on the Pico, and the I2C address (usually 0x3C) |
| "Invalid data" on OLED | The Pico received a name it doesn't know. Make sure the PC and Pico shape lists match |
| Similar shapes confused (e.g. nonagon vs decagon, square vs rhombus) | Draw larger and cleaner, and use a single continuous stroke. Since the training data is synthetic, tuning the generator or training for more epochs can help |
| `pyserial not installed` | `pip install pyserial` |

## Notes

- The OLED shows the shape **name** as large centered text. Long names like `PARALLELOGRAM` are split over two lines.
- Drawing actual outline icons on the OLED can be added later as a separate phase.
