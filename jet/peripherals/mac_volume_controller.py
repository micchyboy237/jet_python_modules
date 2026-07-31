import subprocess
import tkinter as tk


class MacVolumeController:
    def __init__(self, root):
        self.root = root
        self.root.title("Mac System Audio Control")
        self.root.geometry("350x180")
        self.root.resizable(False, False)

        # 1. Get current system volume so the slider starts in the right place
        try:
            result = subprocess.run(
                ["osascript", "-e", "output volume of (get volume settings)"],
                capture_output=True,
                text=True,
                check=True,
            )
            current_vol = int(result.stdout.strip())
        except Exception:
            current_vol = 50  # Fallback if reading fails

        # --- GUI Setup ---
        self.label = tk.Label(
            root, text="System Volume (Active Output)", font=("Arial", 12, "bold")
        )
        self.label.pack(pady=10)

        # 2. Create the slider (0 to 100)
        self.slider = tk.Scale(
            root,
            from_=0,
            to=100,
            orient=tk.HORIZONTAL,
            length=300,
            command=self.update_volume,
            tickinterval=25,
            resolution=1,
        )
        self.slider.pack(padx=20, pady=5)

        # Set slider to the current Mac volume
        self.slider.set(current_vol)

        # Volume percentage label
        self.vol_label = tk.Label(root, text=f"{current_vol}%", font=("Arial", 10))
        self.vol_label.pack()

    def update_volume(self, val):
        """Callback function triggered when the slider is moved."""
        # Use AppleScript to change the macOS master output volume
        subprocess.run(["osascript", "-e", f"set volume output volume {val}"])

        # Update the text label
        self.vol_label.config(text=f"{int(val)}%")


if __name__ == "__main__":
    root = tk.Tk()
    app = MacVolumeController(root)
    root.mainloop()
