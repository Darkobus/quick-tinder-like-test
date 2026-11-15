"""Desktop front-end (CustomTkinter) for matching.py."""
from pathlib import Path
from tkinter import Scrollbar, StringVar, filedialog

import customtkinter as ctk
import pandas as pd

import matching


class PairMatchingApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Pair Matching Tool")
        self.geometry("700x500")

        ctk.set_appearance_mode("system")
        ctk.set_default_color_theme("blue")

        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.left_panel = ctk.CTkFrame(self.main_frame)
        self.left_panel.pack(side="left", fill="y", padx=10)

        self.title_label = ctk.CTkLabel(self.left_panel, text="Pair Matching Tool", font=("Helvetica", 16, "bold"))
        self.title_label.pack(pady=10)

        self.file_button = ctk.CTkButton(self.left_panel, text="Browse Input File", command=self.load_file, fg_color="#007A73")
        self.file_button.pack(pady=5)
        self.file_entry = ctk.CTkEntry(self.left_panel, width=200)
        self.file_entry.pack(pady=5)

        self.output_button = ctk.CTkButton(self.left_panel, text="Browse Output Directory", command=self.select_output_dir, fg_color="#007A73")
        self.output_button.pack(pady=5)
        self.output_entry = ctk.CTkEntry(self.left_panel, width=200)
        self.output_entry.pack(pady=5)

        self.algorithm_label = ctk.CTkLabel(self.left_panel, text="Select Matching Algorithm:")
        self.algorithm_label.pack(pady=(10, 0))

        self.algorithms = list(matching.ALGORITHMS)
        self.algorithm_var = StringVar(value=self.algorithms[0])

        self.algorithm_menu = ctk.CTkOptionMenu(
            self.left_panel,
            variable=self.algorithm_var,
            values=self.algorithms,
            command=self.on_algorithm_change
        )
        self.algorithm_menu.pack(pady=5)

        self.description_frame = ctk.CTkFrame(self.left_panel)
        self.description_frame.pack(fill="x", pady=5)

        self.algorithm_description = ctk.CTkTextbox(self.description_frame, height=80, width=200, wrap="word")
        self.algorithm_description.pack(fill="both", expand=True, padx=5, pady=5)
        self.update_algorithm_description(self.algorithm_var.get())

        self.start_button = ctk.CTkButton(self.left_panel, text="Start Matching", command=self.make_pairs, fg_color="#007A73", font=("Helvetica", 14))
        self.start_button.pack(pady=10)

        self.right_panel = ctk.CTkFrame(self.main_frame)
        self.right_panel.pack(side="right", fill="both", expand=True, padx=10)

        self.results_label = ctk.CTkLabel(self.right_panel, text="Matching Results", font=("Helvetica", 14, "bold"))
        self.results_label.pack(pady=5)

        self.result_frame = ctk.CTkFrame(self.right_panel)
        self.result_frame.pack(fill="both", expand=True)

        self.result_text = ctk.CTkTextbox(self.result_frame, height=150, width=350)
        self.result_text.pack(side="left", fill="both", expand=True)

        self.scrollbar = Scrollbar(self.result_frame, command=self.result_text.yview)
        self.scrollbar.pack(side="right", fill="y")
        self.result_text.configure(yscrollcommand=self.scrollbar.set)

        self.status_bar = ctk.CTkLabel(self, text="Ready", anchor="w")
        self.status_bar.pack(side="bottom", fill="x", padx=10)

        self.file_path = ""
        self.output_dir = ""

    def update_algorithm_description(self, algorithm_name):
        """Update the description text based on the selected algorithm."""
        descriptions = {name: desc for name, (_, desc) in matching.ALGORITHMS.items()}

        self.algorithm_description.delete("1.0", "end")
        self.algorithm_description.insert("1.0", descriptions.get(algorithm_name, "No description available."))

    def on_algorithm_change(self, algorithm_name):
        """Handler for algorithm selection change."""
        self.update_algorithm_description(algorithm_name)

    def load_file(self):
        """Open file dialog to select an Excel file."""
        self.file_path = filedialog.askopenfilename(filetypes=[("Excel Files", "*.xlsx")])
        self.file_entry.delete(0, "end")
        self.file_entry.insert(0, self.file_path)
        if self.file_path:
            self.status_bar.configure(text=f"File loaded: {self.file_path.split('/')[-1]}")

    def select_output_dir(self):
        """Open directory dialog to select output folder."""
        self.output_dir = filedialog.askdirectory()
        self.output_entry.delete(0, "end")
        self.output_entry.insert(0, self.output_dir)
        if self.output_dir:
            self.status_bar.configure(text=f"Output directory set: {self.output_dir.split('/')[-1]}")

    def make_pairs(self):
        """Run the selected algorithm and show / save the pairs."""
        try:
            if not self.file_path:
                raise ValueError("No file selected!")
            self.status_bar.configure(text="Processing...")
            self.update()

            pairs = matching.run(pd.read_excel(self.file_path), self.algorithm_var.get())

            self.result_text.delete("1.0", "end")
            for i, row in enumerate(pairs.itertuples(index=False), 1):
                self.result_text.insert("end", f"Pair {i}: {row[0]} - {row[1]} (Compatibility: {row[2]}%)\n")
            avg = pairs["Compatibility %"].mean() if len(pairs) else 0
            self.result_text.insert("end", f"\nAverage compatibility: {avg:.0f}%\n")

            if self.output_dir:
                output_path = Path(self.output_dir) / "matched_pairs.xlsx"
                pairs.to_excel(output_path, index=False)
                self.status_bar.configure(text=f"Matching complete! Results saved to {output_path.name}")
            else:
                self.status_bar.configure(text="Matching complete!")
        except Exception as e:  # noqa: BLE001 - shown to the user in the window
            self.result_text.delete("1.0", "end")
            self.result_text.insert("end", f"Error: {e}\n")
            self.status_bar.configure(text=f"Error: {e}")


if __name__ == "__main__":
    app = PairMatchingApp()
    app.mainloop()
