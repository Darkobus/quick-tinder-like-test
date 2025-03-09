import pandas as pd
import customtkinter as ctk
from tkinter import filedialog, Scrollbar, StringVar
from itertools import permutations

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
        
        self.algorithm_var = StringVar(value="Basic Greedy")
        self.algorithms = ["Basic Greedy", "Hungarian Algorithm", "Stable Marriage", "Maximum Weight Matching", "K-Means Clustering"]
        
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
        descriptions = {
            "Basic Greedy": "Simple greedy algorithm that matches each person with their best available match based on preference scores.",
            "Hungarian Algorithm": "Optimal assignment algorithm that guarantees the best overall matching for all participants based on compatibility scores.",
            "Stable Marriage": "Ensures no two people would prefer each other over their assigned partners (stable matching).",
            "Maximum Weight Matching": "Network algorithm that maximizes the total compatibility of all pairs.",
            "K-Means Clustering": "Groups similar people together first, then creates pairs within each cluster."
        }
        
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
        """Match pairs based on the selected algorithm."""
        try:
            assert self.file_path, "No file selected!"
            
            self.status_bar.configure(text="Processing...")
            self.update()
            
            df = pd.read_excel(self.file_path)
            df.columns = [col.lower().strip() for col in df.columns]
            
            required_columns = {"Name", "Desired Age Range", "Desired Work Type", "Desired Department"}
            assert required_columns.issubset(df.columns), "Missing required columns in the file!"
      
            df = df.fillna("")
            df["Desired Age Range"] = df["Desired Age Range"].str.lower()
            df["Desired Work Type"] = df["Desired Work Type"].str.lower()
            df["Desired Department"] = df["Desired Department"].str.lower()
            
            algorithm = self.algorithm_var.get()
            
            if algorithm == "Basic Greedy":
                pairs = self.optimize_matching(df)
            elif algorithm == "Hungarian Algorithm":
                pairs = self.hungarian_matching(df)
            elif algorithm == "Stable Marriage":
                pairs = self.stable_marriage_matching(df)
            elif algorithm == "Maximum Weight Matching":
                pairs = self.maximum_weight_matching(df)
            elif algorithm == "K-Means Clustering":
                pairs = self.kmeans_matching(df)
            else:
                pairs = self.optimize_matching(df)
            
            self.result_text.delete("1.0", "end")
            
            for i, (person1, person2) in enumerate(pairs, 1):

                idx1 = df[df["Name"] == person1].index[0]
                idx2 = df[df["Name"] == person2].index[0]
                
                compatibility = sum([
                    df.iloc[idx1]["Desired Age Range"] == df.iloc[idx2]["Desired Age Range"],
                    df.iloc[idx1]["Desired Work Type"] == df.iloc[idx2]["Desired Work Type"],
                    df.iloc[idx1]["Desired Department"] == df.iloc[idx2]["Desired Department"]
                ])
                
                max_compatibility = 3 
                percentage = round((compatibility / max_compatibility) * 100)
                
                self.result_text.insert("end", f"Pair {i}: {person1} - {person2} (Compatibility: {percentage}%)\n")
            
            if self.output_dir:
                output_path = f"{self.output_dir}/matched_pairs.xlsx"
                pd.DataFrame(pairs, columns=["Person A", "Person B"]).to_excel(output_path, index=False)
                self.status_bar.configure(text=f"Matching complete! Results saved to {output_path.split('/')[-1]}")
            else:
                self.status_bar.configure(text="Matching complete!")
                
        except AssertionError as e:
            self.result_text.delete("1.0", "end")
            self.result_text.insert("end", f"Error: {str(e)}\n")
            self.status_bar.configure(text=f"Error: {str(e)}")
        except Exception as e:
            self.result_text.delete("1.0", "end")
            self.result_text.insert("end", f"Unexpected Error: {str(e)}\n")
            self.status_bar.configure(text=f"Error: {str(e)}")

    def optimize_matching(self, df):
        """Basic greedy matching algorithm (original method)."""
        names = df["Name"].tolist()
        best_pairs = []
        used = set()
        
        for i, person in df.iterrows():
            if person["Name"] in used:
                continue
            best_match = None
            best_score = -1
            
            for j, other in df.iterrows():
                if person["Name"] == other["Name"] or other["Name"] in used:
                    continue
                
                score = sum([
                    person["Desired Age Range"] == other["Desired Age Range"],
                    person["Desired Work Type"] == other["Desired Work Type"],
                    person["Desired Department"] == other["Desired Department"]
                ])
                
                if score > best_score:
                    best_match = other["Name"]
                    best_score = score
            
            if best_match:
                best_pairs.append((person["Name"], best_match))
                used.add(person["Name"])
                used.add(best_match)
        
        return best_pairs

    def hungarian_matching(self, df):
        """Hungarian algorithm for optimal assignment."""
        try:
            from scipy.optimize import linear_sum_assignment
            import numpy as np
        except ImportError:
            self.result_text.insert("end", "Error: This matching method requires scipy. Please install it with 'pip install scipy'.\n")
            return self.optimize_matching(df)
        
        n_people = len(df)
        
        if n_people % 2 != 0:

            df = df.iloc[:-1]
            n_people -= 1
        
        cost_matrix = np.zeros((n_people, n_people))
        
        for i in range(n_people):
            for j in range(n_people):
                if i != j:
                    person1 = df.iloc[i]
                    person2 = df.iloc[j]
                    
                    compatibility = sum([
                        person1["Desired Age Range"] == person2["Desired Age Range"],
                        person1["Desired Work Type"] == person2["Desired Work Type"],
                        person1["Desired Department"] == person2["Desired Department"]
                    ])
                    
                    cost_matrix[i][j] = -compatibility
                else:
                    cost_matrix[i][j] = 1000
        
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        pairs = []
        used = set()
        
        for i, j in zip(row_ind, col_ind):
            person1 = df.iloc[i]["Name"]
            person2 = df.iloc[j]["Name"]
            
            if i != j and person1 not in used and person2 not in used:
                pairs.append((person1, person2))
                used.add(person1)
                used.add(person2)
        
        return pairs

    def stable_marriage_matching(self, df):
        """Stable Marriage Algorithm (Gale-Shapley)."""
        names = df["Name"].tolist()
        n = len(names)
        
        if n % 2 != 0:
            n -= 1
            names = names[:n]
        
        preferences = {}
        
        for person_name in names:
            preferences[person_name] = []
            person = df[df["Name"] == person_name].iloc[0]
            
            for other_name in names:
                if person_name != other_name:
                    other = df[df["Name"] == other_name].iloc[0]
                    
                    score = sum([
                        person["Desired Age Range"] == other["Desired Age Range"],
                        person["Desired Work Type"] == other["Desired Work Type"],
                        person["Desired Department"] == other["Desired Department"]
                    ])
                    
                    preferences[person_name].append((score, other_name))
            
            preferences[person_name].sort(reverse=True)

            preferences[person_name] = [Name for _, Name in preferences[person_name]]
        
        group1 = names[:n//2]
        group2 = names[n//2:]
        
        free = group1.copy()

        pairs = {}
        
        next_pref = {person: 0 for person in group1}
        
        while free:

            person = free[0]

            if next_pref[person] >= len(group2):
                free.pop(0)
                continue
                
            next_choice = preferences[person][next_pref[person]]
            next_pref[person] += 1
            

            if next_choice not in pairs:
                pairs[next_choice] = person
                pairs[person] = next_choice
                free.pop(0)
            else:
                current_match = pairs[next_choice]
                

                if preferences[next_choice].index(person) < preferences[next_choice].index(current_match):

                    pairs[next_choice] = person
                    pairs[person] = next_choice

                    pairs.pop(current_match)

                    free.append(current_match)
                    free.pop(0)
        
        result = []
        used = set()
        
        for person1, person2 in pairs.items():
            if person1 not in used and person2 not in used:
                result.append((person1, person2))
                used.add(person1)
                used.add(person2)
        
        return result

    def maximum_weight_matching(self, df):
        """Maximum Weight Matching using NetworkX."""
        try:
            import networkx as nx
        except ImportError:
            self.result_text.insert("end", "Error: This matching method requires networkx. Please install it with 'pip install networkx'.\n")
            return self.optimize_matching(df)  # Fall back to original method
        
        G = nx.Graph()
        
        # Add nodes for each person
        for Name in df["Name"]:
            G.add_node(Name)
        
        # Add edges with weights between compatible pairs
        for i, person1 in df.iterrows():
            for j, person2 in df.iterrows():
                if i < j:  # Avoid duplicates
                    score = sum([ 
                        person1["Desired Age Range"] == person2["Desired Age Range"],
                        person1["Desired Work Type"] == person2["Desired Work Type"],
                        person1["Desired Department"] == person2["Desired Department"]
                    ])
                    
                    # Add edge with weight equal to compatibility score
                    G.add_edge(person1["Name"], person2["Name"], weight=score)
        
        # Find maximum weight matching
        matching = nx.algorithms.matching.max_weight_matching(G, maxcardinality=True, weight="weight")
        
        # Convert the matching result into a list of pairs
        pairs = []
        for person1, person2 in matching:
            pairs.append((person1, person2))
        
        return pairs

    def kmeans_matching(self, df):
        """K-Means Clustering for matching."""
        try:
            from sklearn.cluster import KMeans
            import numpy as np
        except ImportError:
            self.result_text.insert("end", "Error: This matching method requires scikit-learn. Please install it with 'pip install scikit-learn'.\n")
            return self.optimize_matching(df)  # Fall back to original method
        
        # Extract features for clustering
        features = []
        
        # Create a simple encoding for categorical features
        age_range_mapping = {val: i for i, val in enumerate(df["Desired Age Range"].unique())}
        work_type_mapping = {val: i for i, val in enumerate(df["Desired Work Type"].unique())}
        department_mapping = {val: i for i, val in enumerate(df["Desired Department"].unique())}
        
        for _, row in df.iterrows():
            # Convert categorical features to numeric
            age_range = age_range_mapping.get(row["Desired Age Range"], 0)
            work_type = work_type_mapping.get(row["Desired Work Type"], 0)
            department = department_mapping.get(row["Desired Department"], 0)
            
            features.append([age_range, work_type, department])
        
        # Convert to numpy array
        X = np.array(features)
        
        # Determine number of clusters (half the number of people, rounded up)
        n_clusters = max(1, (len(df) + 1) // 2)
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X)
        
        # Add cluster assignments to dataframe
        df_with_clusters = df.copy()
        df_with_clusters["cluster"] = clusters
        
        # Create pairs within clusters
        pairs = []
        used = set()
        
        for cluster_id in range(n_clusters):
            # Get people in this cluster
            cluster_df = df_with_clusters[df_with_clusters["cluster"] == cluster_id]
            
            # Pair people within the cluster using greedy approach
            cluster_people = cluster_df["Name"].tolist()
            
            for i, person1_name in enumerate(cluster_people):
                if person1_name in used:
                    continue
                    
                best_match = None
                best_score = -1
                
                for j, person2_name in enumerate(cluster_people):
                    if i == j or person2_name in used:
                        continue
                        
                    person1 = df[df["Name"] == person1_name].iloc[0]
                    person2 = df[df["Name"] == person2_name].iloc[0]
                    
                    score = sum([
                        person1["Desired Age Range"] == person2["Desired Age Range"],
                        person1["Desired Work Type"] == person2["Desired Work Type"],
                        person1["Desired Department"] == person2["Desired Department"]
                    ])
                    
                    if score > best_score:
                        best_match = person2_name
                        best_score = score
                
                if best_match:
                    pairs.append((person1_name, best_match))
                    used.add(person1_name)
                    used.add(best_match)
        
        # Handle leftover people (across clusters)
        leftover = [Name for Name in df["Name"].tolist() if Name not in used]
        
        for i in range(0, len(leftover)-1, 2):
            if i+1 < len(leftover):
                pairs.append((leftover[i], leftover[i+1]))
        
        return pairs

if __name__ == "__main__":
    app = PairMatchingApp()
    app.mainloop()