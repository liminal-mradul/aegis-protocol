import pandas as pd
from fpdf import FPDF
import os

class CSVtoPDF(FPDF):
    def header(self):
        self.set_font("Arial", 'B', 12)
        self.cell(0, 10, "CSV Data Export", 0, 1, 'C')
        self.ln(5)

    def create_table(self, df, filename):
        self.add_page()
        self.set_font("Arial", 'B', 14)
        self.cell(0, 10, f"File: {filename}", 0, 1, 'L')
        self.ln(5)
        
        # Table Settings
        self.set_font("Arial", size=10)
        line_height = self.font_size * 2.5
        col_width = self.epw / len(df.columns)  # Distribute width equally

        # Header
        self.set_font("Arial", 'B', 10)
        self.set_fill_color(200, 220, 255)
        for col in df.columns:
            self.multi_cell(col_width, line_height, str(col), border=1, 
                            align='C', fill=True, ln=3, max_line_height=self.font_size)
        self.ln(line_height)

        # Data Rows
        self.set_font("Arial", size=10)
        for _, row in df.iterrows():
            for datum in row:
                self.multi_cell(col_width, line_height, str(datum), border=1, 
                                align='L', ln=3, max_line_height=self.font_size)
            self.ln(line_height)

def compile_csvs_to_pdf(folder_path, output_pdf):
    pdf = CSVtoPDF()
    
    # Get all CSV files in the folder
    files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    
    if not files:
        print("No CSV files found in the directory.")
        return

    for file in files:
        file_path = os.path.join(folder_path, file)
        try:
            df = pd.read_csv(file_path)
            pdf.create_table(df, file)
        except Exception as e:
            print(f"Error processing {file}: {e}")

    pdf.output(output_pdf)
    print(f"Successfully created: {output_pdf}")

# --- Configuration ---
folder_directory = input("Enter directory path: ")  # Change this to your folder
output_filename = "Compiled_Report.pdf"

compile_csvs_to_pdf(folder_directory, output_filename)

