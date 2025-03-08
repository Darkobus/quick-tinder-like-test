import pandas as pd
import customtkinter as ctk
from tkinter import filedialog

class PairMatchingApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Pair Matching Tool")
        self.geometry("500x400")
        
        # File selection
        self.file_label = ctk.CTkLabel(self, text="Select Input File:")
        self.file_label.pack(pady=5)
        self.file_button = ctk.CTkButton(self, text="Browse", command=self.load_file)
        self.file_button.pack(pady=5)
        
        # Output directory selection
        self.output_label = ctk.CTkLabel(self, text="Select Output Directory:")
        self.output_label.pack(pady=5)
        self.output_button = ctk.CTkButton(self, text="Browse", command=self.select_output_dir)
        self.output_button.pack(pady=5)
        
        # Start button
        self.start_button = ctk.CTkButton(self, text="Start Matching", command=self.make_pairs, fg_color="#4CAF50")
        self.start_button.pack(pady=10)
        
        # Display matched pairs
        self.result_text = ctk.CTkTextbox(self, height=150, width=400)
        self.result_text.pack(pady=10)
        
        self.file_path = ""
        self.output_dir = ""
    
    def load_file(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("Excel Files", "*.xlsx")])
    
    def select_output_dir(self):
        self.output_dir = filedialog.askdirectory()
    
    def make_pairs(self):
        if not self.file_path:
            self.result_text.insert("end", "No file selected!\n")
            return
        
        df = pd.read_excel(self.file_path)
        pairs = self.optimize_matching(df)
        
        self.result_text.delete("1.0", "end")
        for pair in pairs:
            self.result_text.insert("end", f"{pair[0]} - {pair[1]}\n")
        
        if self.output_dir:
            output_path = f"{self.output_dir}/matched_pairs.xlsx"
            pd.DataFrame(pairs, columns=["Person A", "Person B"]).to_excel(output_path, index=False)
    
    def optimize_matching(self, df):
        names = df["name"].tolist()
        pairs = [(names[i], names[i+1]) for i in range(0, len(names)-1, 2)]
        return pairs
    
if __name__ == "__main__":
    app = PairMatchingApp()
    app.mainloop()