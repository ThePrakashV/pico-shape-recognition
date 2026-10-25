"""
Offline Handwritten SHAPE Recognition - PC Side (single file)
================================================================
19-class geometric shape classifier: Circle, Oval, Triangle, Square,
Rectangle, Parallelogram, Rhombus, Trapezoid, Kite, Pentagon, Hexagon,
Heptagon, Octagon, Nonagon, Decagon, Star, Heart, Crescent, Semicircle.

Contains: synthetic shape-dataset generator + CNN model + training +
preprocessing + prediction + Tkinter drawing GUI + USB serial send to
Raspberry Pi Pico.

Why a synthetic generator instead of a downloaded dataset (unlike the
digit project which used MNIST): no ready-made "hand drawn geometric
shape" dataset exists that matches freehand canvas strokes across all
19 classes, so this script procedurally draws thousands of randomized
outline shapes (random size / rotation / position / stroke width) as
training data. Fully offline, no internet needed at all.

Setup (one-time):
    pip install torch pillow numpy pyserial

Run:
    python shape_cnn_app.py

First run: no saved model exists yet, so the script generates the
synthetic dataset and trains a small CNN, then saves it as
shape_cnn.pth next to this file. Every run after that loads the saved
model instantly.
"""

import os
import math
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import tkinter as tk
from tkinter import messagebox

try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shape_cnn.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_SIZE = 64             # model input resolution (bigger than digit's 28x28
                          # because polygons like nonagon/decagon vs octagon
                          # need more pixels to tell apart)
CANVAS_SIZE = 320         # on-screen canvas, IMG_SIZE * 5 for easy drawing
BRUSH_RADIUS = 8
BAUD_RATE = 9600

# Fixed class order - MUST stay identical between training and inference.
# Also saved inside the checkpoint file itself as a safety net.
SHAPE_CLASSES = [
    "circle", "oval", "triangle", "square", "rectangle",
    "parallelogram", "rhombus", "trapezoid", "kite",
    "pentagon", "hexagon", "heptagon", "octagon", "nonagon", "decagon",
    "star", "heart", "crescent", "semicircle",
]

REGULAR_N = {
    "triangle": 3, "pentagon": 5, "hexagon": 6, "heptagon": 7,
    "octagon": 8, "nonagon": 9, "decagon": 10,
}

PAD = 6  # keep generated shapes away from the image edge


# ======================================================================
# 1. SYNTHETIC SHAPE GENERATOR
#    Draws a randomized OUTLINE (not filled) version of each shape,
#    since a Tkinter canvas pen stroke is an outline, not a filled
#    region. Every shape is first rasterized as a filled mask, then
#    converted to a stroke-width outline via erosion-subtraction -
#    one generic routine that works identically for every class.
# ======================================================================
class ShapeGenerator:
    @staticmethod
    def _rot(points, cx, cy, angle_deg):
        a = math.radians(angle_deg)
        ca, sa = math.cos(a), math.sin(a)
        out = []
        for x, y in points:
            dx, dy = x - cx, y - cy
            out.append((cx + dx * ca - dy * sa, cy + dx * sa + dy * ca))
        return out

    @staticmethod
    def _regular_polygon(cx, cy, r, n, rotation=0.0, start_angle=-90.0):
        pts = []
        for i in range(n):
            ang = math.radians(start_angle + i * 360.0 / n)
            pts.append((cx + r * math.cos(ang), cy + r * math.sin(ang)))
        if rotation:
            pts = ShapeGenerator._rot(pts, cx, cy, rotation)
        return pts

    @staticmethod
    def build_filled_mask(name, size, rng):
        """Returns a PIL 'L' image: 255 = inside shape, 0 = outside."""
        img = Image.new("L", (size, size), 0)
        draw = ImageDraw.Draw(img)
        cx, cy = size / 2, size / 2
        max_r = size / 2 - PAD
        rotation = rng.uniform(0, 360)

        if name == "circle":
            r = rng.uniform(max_r * 0.55, max_r)
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)

        elif name == "oval":
            rx = rng.uniform(max_r * 0.5, max_r)
            ry = rng.uniform(max_r * 0.35, max_r * 0.75)
            temp = Image.new("L", (size, size), 0)
            ImageDraw.Draw(temp).ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=255)
            img = temp.rotate(rotation, center=(cx, cy), resample=Image.BICUBIC)

        elif name in REGULAR_N:
            n = REGULAR_N[name]
            r = rng.uniform(max_r * 0.6, max_r)
            pts = ShapeGenerator._regular_polygon(cx, cy, r, n, rotation=rotation)
            draw.polygon(pts, fill=255)

        elif name == "square":
            s = rng.uniform(max_r * 0.9, max_r * 1.35)
            pts = [(cx - s, cy - s), (cx + s, cy - s), (cx + s, cy + s), (cx - s, cy + s)]
            draw.polygon(ShapeGenerator._rot(pts, cx, cy, rotation), fill=255)

        elif name == "rectangle":
            w = rng.uniform(max_r * 0.6, max_r * 1.3)
            h = rng.uniform(max_r * 0.4, w * 0.75)
            pts = [(cx - w, cy - h), (cx + w, cy - h), (cx + w, cy + h), (cx - w, cy + h)]
            draw.polygon(ShapeGenerator._rot(pts, cx, cy, rotation), fill=255)

        elif name == "parallelogram":
            w = rng.uniform(max_r * 0.6, max_r * 1.1)
            h = rng.uniform(max_r * 0.35, max_r * 0.7)
            shear = rng.uniform(0.3, 0.7) * w
            pts = [(cx - w, cy + h), (cx - w + shear, cy - h), (cx + w, cy - h), (cx + w - shear, cy + h)]
            draw.polygon(ShapeGenerator._rot(pts, cx, cy, rotation), fill=255)

        elif name == "rhombus":
            d1 = rng.uniform(max_r * 1.0, max_r * 1.5)
            d2 = rng.uniform(d1 * 0.35, d1 * 0.65)   # kept clearly != d1 so it
                                                       # doesn't collapse into a square
            pts = [(cx, cy - d1), (cx + d2, cy), (cx, cy + d1), (cx - d2, cy)]
            draw.polygon(ShapeGenerator._rot(pts, cx, cy, rotation), fill=255)

        elif name == "trapezoid":
            top_w = rng.uniform(max_r * 0.3, max_r * 0.7)
            bot_w = rng.uniform(top_w * 1.4, max_r * 1.3)
            h = rng.uniform(max_r * 0.4, max_r * 0.8)
            pts = [(cx - top_w, cy - h), (cx + top_w, cy - h), (cx + bot_w, cy + h), (cx - bot_w, cy + h)]
            draw.polygon(ShapeGenerator._rot(pts, cx, cy, rotation), fill=255)

        elif name == "kite":
            a = rng.uniform(max_r * 0.5, max_r * 0.9)
            c = rng.uniform(a * 1.3, max_r * 1.4)
            b = rng.uniform(max_r * 0.5, max_r * 0.95)
            pts = [(cx, cy - a), (cx + b, cy), (cx, cy + c), (cx - b, cy)]
            draw.polygon(ShapeGenerator._rot(pts, cx, cy, rotation), fill=255)

        elif name == "star":
            n_points = rng.choice([5, 6, 7])
            r_out = rng.uniform(max_r * 0.75, max_r)
            r_in = r_out * rng.uniform(0.38, 0.55)
            pts = []
            for i in range(n_points * 2):
                r = r_out if i % 2 == 0 else r_in
                ang = math.radians(-90 + i * 360.0 / (n_points * 2))
                pts.append((cx + r * math.cos(ang), cy + r * math.sin(ang)))
            draw.polygon(ShapeGenerator._rot(pts, cx, cy, rotation), fill=255)

        elif name == "heart":
            scale = rng.uniform(max_r * 0.055, max_r * 0.07)
            pts = []
            steps = 60
            for i in range(steps):
                t = 2 * math.pi * i / steps
                x = 16 * math.sin(t) ** 3
                y = -(13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t))
                pts.append((cx + x * scale, cy + y * scale))
            draw.polygon(ShapeGenerator._rot(pts, cx, cy, rotation), fill=255)

        elif name == "semicircle":
            r = rng.uniform(max_r * 0.6, max_r)
            bbox = [cx - r, cy - r, cx + r, cy + r]
            draw.pieslice(bbox, start=0, end=180, fill=255)
            img = img.rotate(rotation, center=(cx, cy), resample=Image.BICUBIC)

        elif name == "crescent":
            r = rng.uniform(max_r * 0.65, max_r)
            offset = rng.uniform(r * 0.35, r * 0.65)
            direction = rng.uniform(0, 360)
            dx = offset * math.cos(math.radians(direction))
            dy = offset * math.sin(math.radians(direction))
            big = Image.new("L", (size, size), 0)
            ImageDraw.Draw(big).ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)
            small_r = r * rng.uniform(0.8, 0.95)
            small = Image.new("L", (size, size), 0)
            ImageDraw.Draw(small).ellipse(
                [cx + dx - small_r, cy + dy - small_r, cx + dx + small_r, cy + dy + small_r], fill=255)
            big_arr = np.array(big, dtype=bool)
            small_arr = np.array(small, dtype=bool)
            out_arr = np.logical_and(big_arr, np.logical_not(small_arr))
            img = Image.fromarray((out_arr * 255).astype(np.uint8))

        else:
            raise ValueError(f"Unknown shape: {name}")

        return img

    @staticmethod
    def mask_to_outline(mask_img, stroke_width):
        eroded = mask_img
        for _ in range(stroke_width):
            eroded = eroded.filter(ImageFilter.MinFilter(3))
        arr = np.array(mask_img, dtype=np.int16) - np.array(eroded, dtype=np.int16)
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    @staticmethod
    def generate(name, size, rng, stroke_width=None):
        if stroke_width is None:
            stroke_width = rng.randint(2, 4)  # vary stroke thickness like a real pen
        mask = ShapeGenerator.build_filled_mask(name, size, rng)
        return ShapeGenerator.mask_to_outline(mask, stroke_width)


# ======================================================================
# 2. CNN MODEL (19-class, 64x64 grayscale input)
# ======================================================================
class ShapeCNN(nn.Module):
    def __init__(self, num_classes=len(SHAPE_CLASSES)):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # 64x64 -> 32x32
        x = self.pool(F.relu(self.conv2(x)))   # 32x32 -> 16x16
        x = self.pool(F.relu(self.conv3(x)))   # 16x16 -> 8x8
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        return self.fc2(x)


# ======================================================================
# 3. SYNTHETIC DATASET
#    Built ONCE into memory (uint8 arrays) so training epochs are fast -
#    no per-epoch regeneration cost. Train set uses unseeded randomness
#    (fresh variety), validation set uses a fixed seed (reproducible
#    accuracy tracking across epochs).
# ======================================================================
def build_dataset(shape_names, samples_per_class, img_size, seed, log_fn=print):
    rng = random.Random(seed)
    n_classes = len(shape_names)
    total = n_classes * samples_per_class
    images = np.zeros((total, img_size, img_size), dtype=np.uint8)
    labels = np.zeros((total,), dtype=np.int64)

    idx = 0
    for label, name in enumerate(shape_names):
        for _ in range(samples_per_class):
            img = ShapeGenerator.generate(name, img_size, rng)
            images[idx] = np.array(img, dtype=np.uint8)
            labels[idx] = label
            idx += 1
        log_fn(f"  generated {name} ({label + 1}/{n_classes})")
    return images, labels


class ShapeDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        arr = self.images[idx].astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).unsqueeze(0)  # [1, H, W]
        return tensor, int(self.labels[idx])


# ======================================================================
# 4. TRAINING
# ======================================================================
def train_and_save_model(epochs=12, samples_per_class=900, val_per_class=120, log_fn=print):
    log_fn("Generating synthetic training shapes (this only happens once)...")
    train_imgs, train_labels = build_dataset(SHAPE_CLASSES, samples_per_class, IMG_SIZE, seed=1, log_fn=log_fn)
    log_fn("Generating fixed validation set...")
    val_imgs, val_labels = build_dataset(SHAPE_CLASSES, val_per_class, IMG_SIZE, seed=999, log_fn=log_fn)

    train_loader = DataLoader(ShapeDataset(train_imgs, train_labels), batch_size=128, shuffle=True)
    val_loader = DataLoader(ShapeDataset(val_imgs, val_labels), batch_size=256, shuffle=False)

    model = ShapeCNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_acc = 100 * correct / total
        val_acc = evaluate(model, val_loader)
        log_fn(f"Epoch {epoch}/{epochs} - loss: {running_loss/total:.4f} "
               f"- train_acc: {train_acc:.2f}% - val_acc: {val_acc:.2f}%")

    torch.save({"model": model.state_dict(), "classes": SHAPE_CLASSES}, MODEL_PATH)
    log_fn(f"Model saved to {MODEL_PATH}")
    return model


def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total


def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    saved_classes = checkpoint.get("classes", SHAPE_CLASSES)
    if saved_classes != SHAPE_CLASSES:
        raise RuntimeError(
            "Saved model's class order does not match SHAPE_CLASSES in this "
            "script. Delete shape_cnn.pth and retrain, or fix the class list."
        )
    model = ShapeCNN(num_classes=len(saved_classes)).to(DEVICE)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


# ======================================================================
# 5. PREPROCESSING - must match the training representation: white
#    outline on black background, centered in a 64x64 frame with a
#    small margin (same convention as the synthetic generator output).
# ======================================================================
def preprocess_canvas_image(pil_image):
    """pil_image: black strokes drawn on a white Tkinter canvas."""
    img = pil_image.convert("L")
    img = ImageOps.invert(img)  # white strokes on black bg, like training data

    bbox = img.getbbox()
    if bbox is not None:
        img = img.crop(bbox)

    # Keep aspect ratio, fit inside ~46x46 (matches the margin ratio used
    # in the digit project: 20/28), then paste centered onto IMG_SIZE canvas
    margin_box = int(IMG_SIZE * 20 / 28)
    img.thumbnail((margin_box, margin_box), Image.LANCZOS)
    new_img = Image.new("L", (IMG_SIZE, IMG_SIZE), color=0)
    upper_left = ((IMG_SIZE - img.width) // 2, (IMG_SIZE - img.height) // 2)
    new_img.paste(img, upper_left)

    arr = np.array(new_img, dtype=np.float32) / 255.0
    tensor = torch.tensor(arr).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    return tensor


def predict_shape(model, pil_image, topk=3):
    tensor = preprocess_canvas_image(pil_image).to(DEVICE)
    with torch.no_grad():
        outputs = model(tensor)
        probs = F.softmax(outputs, dim=1)[0]
        top_probs, top_idx = torch.topk(probs, k=min(topk, len(SHAPE_CLASSES)))
    results = [(SHAPE_CLASSES[i], float(p) * 100) for p, i in zip(top_probs, top_idx)]
    return results  # sorted best-first, e.g. [("star", 91.2), ("pentagon", 4.1), ...]


# ======================================================================
# 6. SERIAL COMMUNICATION TO PICO
# ======================================================================
class SerialManager:
    def __init__(self):
        self.ser = None
        self.port = None

    @staticmethod
    def list_ports():
        if not SERIAL_AVAILABLE:
            return []
        return [p.device for p in serial.tools.list_ports.comports()]

    def connect(self, port, baud=BAUD_RATE, timeout=2):
        if not SERIAL_AVAILABLE:
            raise RuntimeError("pyserial not installed. Run: pip install pyserial")
        if self.ser and self.ser.is_open:
            self.ser.close()
        self.ser = serial.Serial(port, baud, timeout=timeout)
        self.port = port
        return True

    def is_connected(self):
        return self.ser is not None and self.ser.is_open

    def send_shape(self, name):
        if not self.is_connected():
            raise RuntimeError("Not connected to Pico")
        if name not in SHAPE_CLASSES:
            raise ValueError(f"Invalid shape: {name}")
        self.ser.write(f"{name}\n".encode("utf-8"))

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()


# ======================================================================
# 7. TKINTER GUI
# ======================================================================
class ShapeApp:
    def __init__(self, root, model):
        self.root = root
        self.model = model
        self.serial_mgr = SerialManager()
        root.title("Offline Shape Recognition - PC to Pico")
        root.resizable(False, False)

        tk.Label(root, text="DRAW A SHAPE", font=("Arial", 22, "bold")).pack(pady=(10, 0))
        self.canvas = tk.Canvas(root, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="white", cursor="cross")
        self.canvas.pack(padx=10, pady=10)
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset_stroke)

        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=255)
        self.draw = ImageDraw.Draw(self.image)
        self.last_x, self.last_y = None, None

        self.result_var = tk.StringVar(value="Prediction: -")
        self.top3_var = tk.StringVar(value="")
        tk.Label(root, textvariable=self.result_var, font=("Arial", 16, "bold")).pack()
        tk.Label(root, textvariable=self.top3_var, font=("Arial", 11), justify="left").pack(pady=(0, 10))

        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=5)
        tk.Button(btn_frame, text="CLEAR", width=12, command=self.clear_canvas).grid(row=0, column=0, padx=5)
        tk.Button(btn_frame, text="PREDICT", width=12, command=self.predict).grid(row=0, column=1, padx=5)

        serial_frame = tk.LabelFrame(root, text="Pico Serial Connection")
        serial_frame.pack(padx=10, pady=10, fill="x")

        self.port_var = tk.StringVar()
        ports = self.serial_mgr.list_ports()
        self.port_menu = tk.OptionMenu(serial_frame, self.port_var, *(ports or ["No ports found"]))
        self.port_menu.grid(row=0, column=0, padx=5, pady=5)
        if ports:
            self.port_var.set(ports[0])

        tk.Button(serial_frame, text="Refresh", command=self.refresh_ports).grid(row=0, column=1, padx=5)
        tk.Button(serial_frame, text="Connect", command=self.connect_serial).grid(row=0, column=2, padx=5)

        self.status_var = tk.StringVar(value="Status: Not connected")
        tk.Label(serial_frame, textvariable=self.status_var, fg="red").grid(row=1, column=0, columnspan=3, pady=(0, 5))

    # ---- drawing ----
    def paint(self, event):
        x, y = event.x, event.y
        r = BRUSH_RADIUS
        if self.last_x is not None:
            self.canvas.create_line(self.last_x, self.last_y, x, y, width=r * 2,
                                     fill="black", capstyle=tk.ROUND, smooth=True)
            self.draw.line([self.last_x, self.last_y, x, y], fill=0, width=r * 2)
        else:
            self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="black")
            self.draw.ellipse([x - r, y - r, x + r, y + r], fill=0)
        self.last_x, self.last_y = x, y

    def reset_stroke(self, event):
        self.last_x, self.last_y = None, None

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=255)
        self.draw = ImageDraw.Draw(self.image)
        self.result_var.set("Prediction: -")
        self.top3_var.set("")

    # ---- serial ----
    def refresh_ports(self):
        ports = self.serial_mgr.list_ports()
        menu = self.port_menu["menu"]
        menu.delete(0, "end")
        for p in ports:
            menu.add_command(label=p, command=lambda v=p: self.port_var.set(v))
        if ports:
            self.port_var.set(ports[0])

    def connect_serial(self):
        port = self.port_var.get()
        if not port or port == "No ports found":
            messagebox.showerror("Serial Error", "No COM port selected.")
            return
        try:
            self.serial_mgr.connect(port)
            self.status_var.set(f"Status: Connected to {port}")
        except Exception as e:
            self.status_var.set("Status: Connection failed")
            messagebox.showerror("Serial Error", f"Could not connect to {port}:\n{e}")

    # ---- prediction ----
    def predict(self):
        results = predict_shape(self.model, self.image, topk=3)
        best_name, best_conf = results[0]
        self.result_var.set(f"Prediction: {best_name.upper()} ({best_conf:.1f}%)")
        others = "\n".join(f"  {i+1}. {name} - {conf:.1f}%" for i, (name, conf) in enumerate(results))
        self.top3_var.set(others)

        if not self.serial_mgr.is_connected():
            messagebox.showwarning("Not sent", "Predicted, but Pico is not connected.\n"
                                                "Connect a COM port first.")
            return
        try:
            self.serial_mgr.send_shape(best_name)
        except Exception as e:
            messagebox.showerror("Serial Error", f"Failed to send shape:\n{e}")


# ======================================================================
# 8. MAIN
# ======================================================================
def main():
    if not os.path.exists(MODEL_PATH):
        print("No saved model found - generating synthetic shapes and training a new CNN (first run only)...")
        model = train_and_save_model(epochs=12, samples_per_class=900, val_per_class=120)
    else:
        print(f"Loading saved model from {MODEL_PATH}")
        model = load_model()

    root = tk.Tk()
    app = ShapeApp(root, model)
    root.mainloop()
    app.serial_mgr.close()


if __name__ == "__main__":
    main()
