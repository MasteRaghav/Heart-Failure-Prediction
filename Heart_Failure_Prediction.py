import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import joblib
import os

class HeartFailurePredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Heart Failure Prediction System")
        self.root.geometry("1200x700")
        self.root.configure(bg="#f0f0f0")
        
        self.dataset = None
        self.model = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.predictions = None
        self.scaler = None
        
        # Create main frame
        main_frame = tk.Frame(root, bg="#f0f0f0")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel for controls
        left_panel = tk.LabelFrame(main_frame, text="Controls", bg="#f0f0f0", padx=10, pady=10)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # Right panel for visualizations
        self.right_panel = tk.LabelFrame(main_frame, text="Visualizations", bg="#f0f0f0", padx=10, pady=10)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Control buttons
        tk.Button(left_panel, text="Load Dataset", command=self.load_dataset, width=20, bg="#4CAF50", fg="white").pack(pady=5)
        tk.Button(left_panel, text="Display Data Preview", command=self.display_data_preview, width=20, bg="#2196F3", fg="white").pack(pady=5)
        tk.Button(left_panel, text="Train Model", command=self.train_model, width=20, bg="#FF9800", fg="white").pack(pady=5)
        tk.Button(left_panel, text="Evaluate Model", command=self.evaluate_model, width=20, bg="#9C27B0", fg="white").pack(pady=5)
        tk.Button(left_panel, text="Show Feature Importance", command=self.show_feature_importance, width=20, bg="#607D8B", fg="white").pack(pady=5)
        tk.Button(left_panel, text="Single Prediction", command=self.open_prediction_window, width=20, bg="#E91E63", fg="white").pack(pady=5)
        tk.Button(left_panel, text="Save Model", command=self.save_model, width=20, bg="#795548", fg="white").pack(pady=5)
        tk.Button(left_panel, text="Load Model", command=self.load_model, width=20, bg="#009688", fg="white").pack(pady=5)
        
        # Status frame
        status_frame = tk.LabelFrame(left_panel, text="Status", bg="#f0f0f0", padx=5, pady=5)
        status_frame.pack(fill=tk.X, pady=10)
        
        self.status_label = tk.Label(status_frame, text="Ready", bg="#f0f0f0", wraplength=200)
        self.status_label.pack(fill=tk.X)
        
        # Dataset info frame
        info_frame = tk.LabelFrame(left_panel, text="Dataset Info", bg="#f0f0f0", padx=5, pady=5)
        info_frame.pack(fill=tk.X, pady=10)
        
        self.info_label = tk.Label(info_frame, text="No dataset loaded", bg="#f0f0f0", wraplength=200, justify=tk.LEFT)
        self.info_label.pack(fill=tk.X)
        
        # Performance frame
        performance_frame = tk.LabelFrame(left_panel, text="Model Performance", bg="#f0f0f0", padx=5, pady=5)
        performance_frame.pack(fill=tk.X, pady=10)
        
        self.performance_label = tk.Label(performance_frame, text="No model trained", bg="#f0f0f0", wraplength=200, justify=tk.LEFT)
        self.performance_label.pack(fill=tk.X)
        
    def set_status(self, message):
        self.status_label.config(text=message)
        self.root.update()
    
    def load_dataset(self):
        try:
            filename = filedialog.askopenfilename(
                title="Select Dataset",
                filetypes=(("CSV files", "*.csv"), ("Excel files", "*.xlsx"), ("All files", "*.*"))
            )
            
            if not filename:
                return
            
            if filename.endswith('.csv'):
                self.dataset = pd.read_csv(filename)
            elif filename.endswith('.xlsx'):
                self.dataset = pd.read_excel(filename)
            else:
                messagebox.showerror("Error", "Unsupported file format!")
                return
            
            self.set_status(f"Dataset loaded: {os.path.basename(filename)}")
            
            # Display basic dataset information
            self.info_label.config(text=f"Rows: {self.dataset.shape[0]}\nColumns: {self.dataset.shape[1]}\nTarget: {'DEATH_EVENT' if 'DEATH_EVENT' in self.dataset.columns else 'Unknown'}")
            
            # Check if dataset has the expected target column
            if 'DEATH_EVENT' not in self.dataset.columns:
                messagebox.showwarning("Warning", "Dataset may not be compatible. Expected 'DEATH_EVENT' column not found.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error loading dataset: {str(e)}")
            self.set_status("Error loading dataset")
    
    def display_data_preview(self):
        if self.dataset is None:
            messagebox.showerror("Error", "Please load a dataset first!")
            return
        
        # Clear previous visualizations
        for widget in self.right_panel.winfo_children():
            widget.destroy()
        
        # Create a frame for the table
        table_frame = tk.Frame(self.right_panel)
        table_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create a treeview to display the data
        columns = list(self.dataset.columns)
        tree = ttk.Treeview(table_frame, columns=columns, show="headings")
        
        # Add scrollbars
        vsb = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
        hsb = ttk.Scrollbar(table_frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        
        # Pack scrollbars
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        hsb.pack(side=tk.BOTTOM, fill=tk.X)
        tree.pack(fill=tk.BOTH, expand=True)
        
        # Set column headings
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=100, anchor=tk.CENTER)
        
        # Add data rows
        for i, row in self.dataset.head(50).iterrows():
            tree.insert("", tk.END, values=list(row))
        
        self.set_status("Displaying data preview (first 50 rows)")
    
    def train_model(self):
        if self.dataset is None:
            messagebox.showerror("Error", "Please load a dataset first!")
            return
        
        try:
            # Check if target column exists
            if 'DEATH_EVENT' not in self.dataset.columns:
                messagebox.showerror("Error", "Dataset must have 'DEATH_EVENT' column!")
                return
            
            # Prepare data
            X = self.dataset.drop('DEATH_EVENT', axis=1)
            y = self.dataset['DEATH_EVENT']
            
            # Split data
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )
            
            # Scale features
            self.scaler = StandardScaler()
            self.X_train_scaled = self.scaler.fit_transform(self.X_train)
            self.X_test_scaled = self.scaler.transform(self.X_test)
            
            # Train model
            self.set_status("Training model...")
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(self.X_train_scaled, self.y_train)
            
            # Make predictions
            self.predictions = self.model.predict(self.X_test_scaled)
            accuracy = accuracy_score(self.y_test, self.predictions)
            
            self.set_status("Model trained successfully!")
            self.performance_label.config(text=f"Accuracy: {accuracy:.2%}")
            
            messagebox.showinfo("Success", f"Model trained successfully!\nAccuracy: {accuracy:.2%}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error training model: {str(e)}")
            self.set_status("Error training model")
    
    def evaluate_model(self):
        if self.model is None or self.predictions is None:
            messagebox.showerror("Error", "Please train a model first!")
            return
        
        # Clear previous visualizations
        for widget in self.right_panel.winfo_children():
            widget.destroy()
        
        # Create a notebook for tabs
        notebook = ttk.Notebook(self.right_panel)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab for confusion matrix
        cm_tab = ttk.Frame(notebook)
        notebook.add(cm_tab, text="Confusion Matrix")
        
        # Tab for classification report
        cr_tab = ttk.Frame(notebook)
        notebook.add(cr_tab, text="Classification Report")
        
        # Tab for ROC curve
        roc_tab = ttk.Frame(notebook)
        notebook.add(roc_tab, text="ROC Curve")
        
        # Create confusion matrix figure
        cm_fig = Figure(figsize=(6, 5))
        cm_ax = cm_fig.add_subplot(111)
        cm = confusion_matrix(self.y_test, self.predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=cm_ax)
        cm_ax.set_title('Confusion Matrix')
        cm_ax.set_xlabel('Predicted Labels')
        cm_ax.set_ylabel('True Labels')
        
        cm_canvas = FigureCanvasTkAgg(cm_fig, cm_tab)
        cm_canvas.draw()
        cm_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Create classification report text
        cr_frame = tk.Frame(cr_tab)
        cr_frame.pack(fill=tk.BOTH, expand=True)
        
        cr_text = tk.Text(cr_frame, wrap=tk.WORD)
        cr_text.pack(fill=tk.BOTH, expand=True)
        
        cr = classification_report(self.y_test, self.predictions)
        cr_text.insert(tk.END, cr)
        cr_text.config(state=tk.DISABLED)
        
        # Create ROC curve (if possible)
        try:
            from sklearn.metrics import roc_curve, auc
            
            # Get probabilities for the positive class
            y_probs = self.model.predict_proba(self.X_test_scaled)[:, 1]
            
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(self.y_test, y_probs)
            roc_auc = auc(fpr, tpr)
            
            # Create figure
            roc_fig = Figure(figsize=(6, 5))
            roc_ax = roc_fig.add_subplot(111)
            
            # Plot ROC curve
            roc_ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            roc_ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            roc_ax.set_xlim([0.0, 1.0])
            roc_ax.set_ylim([0.0, 1.05])
            roc_ax.set_xlabel('False Positive Rate')
            roc_ax.set_ylabel('True Positive Rate')
            roc_ax.set_title('Receiver Operating Characteristic')
            roc_ax.legend(loc="lower right")
            
            roc_canvas = FigureCanvasTkAgg(roc_fig, roc_tab)
            roc_canvas.draw()
            roc_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
        except Exception as e:
            # Create a label to display the error
            tk.Label(roc_tab, text=f"Could not generate ROC curve: {str(e)}").pack(padx=10, pady=10)
        
        self.set_status("Model evaluation displayed")
    
    def show_feature_importance(self):
        if self.model is None:
            messagebox.showerror("Error", "Please train a model first!")
            return
        
        # Clear previous visualizations
        for widget in self.right_panel.winfo_children():
            widget.destroy()
        
        # Get feature importance
        importances = self.model.feature_importances_
        feature_names = self.X_train.columns
        
        # Sort features by importance
        indices = np.argsort(importances)[::-1]
        
        # Create figure
        fig = Figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        
        # Plot feature importance
        ax.barh(range(len(indices)), importances[indices], color='skyblue')
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_xlabel('Feature Importance')
        ax.set_title('Feature Importance in the Model')
        
        canvas = FigureCanvasTkAgg(fig, self.right_panel)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.set_status("Feature importance displayed")
    
    def open_prediction_window(self):
        if self.model is None or self.scaler is None:
            messagebox.showerror("Error", "Please train a model first!")
            return
            
        # Create prediction window
        pred_window = tk.Toplevel(self.root)
        pred_window.title("Make Single Prediction")
        pred_window.geometry("600x500")
        pred_window.configure(bg="#f0f0f0")
        
        # Create a frame for inputs
        input_frame = tk.Frame(pred_window, bg="#f0f0f0")
        input_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create scrollable canvas
        canvas = tk.Canvas(input_frame, bg="#f0f0f0")
        scrollbar = ttk.Scrollbar(input_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Feature names and entry fields
        feature_entries = {}
        feature_names = self.X_train.columns
        
        # Get feature statistics for defaults
        feature_means = self.X_train.mean()
        
        # Create entries for all features
        for i, feature in enumerate(feature_names):
            row = tk.Frame(scrollable_frame, bg="#f0f0f0")
            row.pack(fill=tk.X, padx=5, pady=5)
            
            tk.Label(row, text=f"{feature}:", width=20, anchor="w", bg="#f0f0f0").pack(side=tk.LEFT)
            entry = tk.Entry(row, width=15)
            entry.pack(side=tk.LEFT, padx=5)
            entry.insert(0, str(round(feature_means[feature], 2)))
            
            feature_entries[feature] = entry
        
        # Function to make prediction
        def make_prediction():
            try:
                # Get values from entries
                input_data = {}
                for feature, entry in feature_entries.items():
                    try:
                        value = float(entry.get())
                        input_data[feature] = value
                    except ValueError:
                        messagebox.showerror("Error", f"Invalid value for {feature}. Please enter a number.")
                        return
                
                # Convert to DataFrame
                input_df = pd.DataFrame([input_data])
                
                # Scale data
                input_scaled = self.scaler.transform(input_df)
                
                # Make prediction
                prediction = self.model.predict(input_scaled)[0]
                probability = self.model.predict_proba(input_scaled)[0][1]
                
                # Display result
                result_text = f"Prediction: {'Heart Failure' if prediction == 1 else 'No Heart Failure'}\n"
                result_text += f"Probability of Heart Failure: {probability:.2%}"
                
                result_label.config(text=result_text)
                
            except Exception as e:
                messagebox.showerror("Error", f"Error making prediction: {str(e)}")
        
        # Add button to make prediction
        btn_frame = tk.Frame(pred_window, bg="#f0f0f0")
        btn_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Button(btn_frame, text="Make Prediction", command=make_prediction, bg="#4CAF50", fg="white").pack()
        
        # Result label
        result_frame = tk.LabelFrame(pred_window, text="Prediction Result", bg="#f0f0f0", padx=10, pady=10)
        result_frame.pack(fill=tk.X, padx=10, pady=10)
        
        result_label = tk.Label(result_frame, text="", bg="#f0f0f0", font=("Arial", 12))
        result_label.pack()
    
    def save_model(self):
        if self.model is None:
            messagebox.showerror("Error", "Please train a model first!")
            return
        
        try:
            # Get save location
            filename = filedialog.asksaveasfilename(
                title="Save Model",
                filetypes=(("Joblib files", "*.joblib"), ("All files", "*.*")),
                defaultextension=".joblib"
            )
            
            if not filename:
                return
            
            # Save model and scaler
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': list(self.X_train.columns)
            }
            
            joblib.dump(model_data, filename)
            
            self.set_status(f"Model saved to {os.path.basename(filename)}")
            messagebox.showinfo("Success", "Model saved successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error saving model: {str(e)}")
            self.set_status("Error saving model")
    
    def load_model(self):
        try:
            # Get file location
            filename = filedialog.askopenfilename(
                title="Load Model",
                filetypes=(("Joblib files", "*.joblib"), ("All files", "*.*"))
            )
            
            if not filename:
                return
            
            # Load model and scaler
            model_data = joblib.load(filename)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            
            self.set_status(f"Model loaded from {os.path.basename(filename)}")
            messagebox.showinfo("Success", "Model loaded successfully!")
            
            # Update performance label
            self.performance_label.config(text="Model loaded from file")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error loading model: {str(e)}")
            self.set_status("Error loading model")

if __name__ == "__main__":
    root = tk.Tk()
    app = HeartFailurePredictionApp(root)
    root.mainloop()