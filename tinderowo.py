import pandas as pd
import customtkinter as ctk
from tkinter import filedialog, Scrollbar
from itertools import permutations

class PairMatchingApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Pair Matching Tool")
        self.geometry("600x400")
        
        # Main frame
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Left panel for buttons and file paths
        self.left_panel = ctk.CTkFrame(self.main_frame)
        self.left_panel.pack(side="left", fill="y", padx=10)
        
        # File selection
        self.file_button = ctk.CTkButton(self.left_panel, text="Browse Input File", command=self.load_file)
        self.file_button.pack(pady=5)
        self.file_entry = ctk.CTkEntry(self.left_panel, width=200)
        self.file_entry.pack(pady=5)
        
        # Output directory selection
        self.output_button = ctk.CTkButton(self.left_panel, text="Browse Output Directory", command=self.select_output_dir)
        self.output_button.pack(pady=5)
        self.output_entry = ctk.CTkEntry(self.left_panel, width=200)
        self.output_entry.pack(pady=5)
        
        # Start button in the center
        self.start_button = ctk.CTkButton(self.main_frame, text="Start Matching", command=self.make_pairs, fg_color="#4CAF50")
        self.start_button.pack(side="bottom", pady=10)
        
        # Right panel for results
        self.right_panel = ctk.CTkFrame(self.main_frame)
        self.right_panel.pack(side="right", fill="both", expand=True, padx=10)
        
        # Scrollable result display
        self.result_frame = ctk.CTkFrame(self.right_panel)
        self.result_frame.pack(fill="both", expand=True)
        
        self.result_text = ctk.CTkTextbox(self.result_frame, height=150, width=250)
        self.result_text.pack(side="left", fill="both", expand=True)
        
        self.scrollbar = Scrollbar(self.result_frame, command=self.result_text.yview)
        self.scrollbar.pack(side="right", fill="y")
        self.result_text.configure(yscrollcommand=self.scrollbar.set)
        
        self.file_path = ""
        self.output_dir = ""
    
    def load_file(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("Excel Files", "*.xlsx")])
        self.file_entry.delete(0, "end")
        self.file_entry.insert(0, self.file_path)
    
    def select_output_dir(self):
        self.output_dir = filedialog.askdirectory()
        self.output_entry.delete(0, "end")
        self.output_entry.insert(0, self.output_dir)
    
    def make_pairs(self):
        try:
            assert self.file_path, "No file selected!"
            
            df = pd.read_excel(self.file_path)
            df.columns = [col.lower().strip() for col in df.columns]  # Ignore case and strip spaces
            
            required_columns = {"name", "desired age range", "desired work type", "desired departament"}
            assert required_columns.issubset(df.columns), "Missing required columns in the file!"
            
            pairs = self.optimize_matching(df)
            
            self.result_text.delete("1.0", "end")
            for pair in pairs:
                self.result_text.insert("end", f"{pair[0]} - {pair[1]}\n")
            
            if self.output_dir:
                output_path = f"{self.output_dir}/matched_pairs.xlsx"
                pd.DataFrame(pairs, columns=["Person A", "Person B"]).to_excel(output_path, index=False)
        except AssertionError as e:
            self.result_text.delete("1.0", "end")
            self.result_text.insert("end", f"Error: {str(e)}\n")
        except Exception as e:
            self.result_text.delete("1.0", "end")
            self.result_text.insert("end", f"Unexpected Error: {str(e)}\n")
    
    def optimize_matching(self, df):
        df = df.fillna("")  # Fill NaN with empty strings
        df["desired age range"] = df["desired age range"].str.lower()
        df["desired work type"] = df["desired work type"].str.lower()
        df["desired departament"] = df["desired departament"].str.lower()
        
        names = df["name"].tolist()
        best_pairs = []
        used = set()
        
        for i, person in df.iterrows():
            if person["name"] in used:
                continue
            best_match = None
            best_score = -1
            
            for j, other in df.iterrows():
                if person["name"] == other["name"] or other["name"] in used:
                    continue
                
                score = sum([
                    person["desired age range"] == other["desired age range"],
                    person["desired work type"] == other["desired work type"],
                    person["desired departament"] == other["desired departament"]
                ])
                
                if score > best_score:
                    best_match = other["name"]
                    best_score = score
            
            if best_match:
                best_pairs.append((person["name"], best_match))
                used.add(person["name"])
                used.add(best_match)
        
        return best_pairs
    
if __name__ == "__main__":
    app = PairMatchingApp()
    app.mainloop()
