import os
import json
import re
from pathlib import Path
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import argparse
from decimal import Decimal
from openpyxl import load_workbook
from openpyxl.utils import get_column_letter
from openpyxl.styles import Alignment
import numpy as np

class SecurityMetricsConsolidator:
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.data = []
        self.method_groups = defaultdict(list)
        self.details_data = []

    def parse_metrics_file(self, filepath: Path) -> Dict:
        """Parse a SALLM_metrics.txt file and extract metrics."""
        with open(filepath, 'r') as f:
            content = f.read()
        
        metrics = {}
        
        # Extract run info
        dataset_match = re.search(r'dataset:\s+(\S+)', content)
        model_match = re.search(r'model:\s+(\S+)', content)
        
        if dataset_match:
            metrics['dataset'] = dataset_match.group(1)
        if model_match:
            metrics['model'] = model_match.group(1)
        
        # Extract token usage
        prompt_tokens = re.search(r'prompt_tokens:\s+(\d+)', content)
        completion_tokens = re.search(r'completion_tokens:\s+(\d+)', content)
        total_tokens = re.search(r'total_tokens:\s+(\d+)', content)
        
        if prompt_tokens:
            metrics['prompt_tokens'] = int(prompt_tokens.group(1))
        if completion_tokens:
            metrics['completion_tokens'] = int(completion_tokens.group(1))
        if total_tokens:
            metrics['total_tokens'] = int(total_tokens.group(1))
        
        # Extract runtime
        runtime = re.search(r'runtime_seconds:\s+([\d.]+)', content)
        if runtime:
            metrics['runtime_seconds'] = float(runtime.group(1))
        
        # Extract Bandit metrics
        bandit_section = re.search(r'--- Bandit ---(.+?)(?=--- Semgrep ---|$)', content, re.DOTALL)
        if bandit_section:
            bandit_text = bandit_section.group(1)
            
            n_match = re.search(r'items \(N\):\s+(\d+)', bandit_text)
            v_match = re.search(r'total issues \(V\):\s+(\d+)', bandit_text)
            l_match = re.search(r'total LOC \(L\):\s+(\d+)', bandit_text)
            
            # Extract rate and density from the calculation lines
            rate_match = re.search(r'Vulnerability Rate = V/N = \d+/\d+ = ([\d.]+)', bandit_text)
            density_match = re.search(r'Vulnerability Density = V/L = \d+/\d+ = ([\d.]+)', bandit_text)
            
            if n_match:
                metrics['bandit_items'] = int(n_match.group(1))
            if v_match:
                metrics['bandit_issues'] = int(v_match.group(1))
            if l_match:
                metrics['bandit_loc'] = int(l_match.group(1))
            if rate_match:
                metrics['bandit_rate'] = float(rate_match.group(1))
            if density_match:
                metrics['bandit_density'] = float(density_match.group(1))
        
        # Extract Semgrep metrics
        semgrep_section = re.search(r'--- Semgrep ---(.+?)$', content, re.DOTALL)
        if semgrep_section:
            semgrep_text = semgrep_section.group(1)
            
            n_match = re.search(r'items \(N\):\s+(\d+)', semgrep_text)
            v_match = re.search(r'total issues \(V\):\s+(\d+)', semgrep_text)
            l_match = re.search(r'total LOC \(L\):\s+(\d+)', semgrep_text)
            
            # Extract rate and density from the calculation lines
            rate_match = re.search(r'Vulnerability Rate = V/N = \d+/\d+ = ([\d.]+)', semgrep_text)
            density_match = re.search(r'Vulnerability Density = V/L = \d+/\d+ = ([\d.]+)', semgrep_text)
            
            if n_match:
                metrics['semgrep_items'] = int(n_match.group(1))
            if v_match:
                metrics['semgrep_issues'] = int(v_match.group(1))
            if l_match:
                metrics['semgrep_loc'] = int(l_match.group(1))
            if rate_match:
                metrics['semgrep_rate'] = float(rate_match.group(1))
            if density_match:
                metrics['semgrep_density'] = float(density_match.group(1))
                        
        return metrics

    def extract_method_info(self, method_dir_name: str) -> Tuple[str, int]:
        """Extract method name and run number from directory name."""
        # Handle cases like 'direct(2)' - extract run number in parentheses
        paren_match = re.search(r'\((\d+)\)$', method_dir_name)
        if paren_match:
            run_num = int(paren_match.group(1))
            method = method_dir_name[:paren_match.start()]
            return method, run_num
        
        # For all other cases, keep the full name as the method
        # This preserves rci_iter1, self_refine_iter2, etc.
        return method_dir_name, 1
    
    def collect_data(self):
        """Traverse the input directory and collect all metrics."""
        for db_dir in self.input_dir.iterdir():
            if not db_dir.is_dir():
                continue
            
            database = db_dir.name
            
            for model_dir in db_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                
                model = model_dir.name
                
                for method_dir in model_dir.iterdir():
                    if not method_dir.is_dir():
                        continue
                    
                    method_name, run_num = self.extract_method_info(method_dir.name)
                    
                    # Handle special case for one_shot directory structure
                    if method_name == 'one_shot':
                        for sub_method in ['one_shot', 'zero_shot']:
                            metrics_file = method_dir / sub_method / 'rate' / 'vuln_density' / f'{sub_method}.txt'
                            if metrics_file.exists():
                                metrics = self.parse_metrics_file(metrics_file)
                                metrics['database'] = database
                                metrics['model'] = model
                                metrics['method'] = sub_method
                                metrics['run_number'] = run_num
                                metrics['method_dir'] = method_dir.name
                                self.data.append(metrics)
                                self.method_groups[sub_method].append(metrics)
                            
                            # Collect details_result.xlsx
                            details_file = method_dir / sub_method / 'details_result' / 'details_result.xlsx'
                            if details_file.exists():
                                self.collect_details_result(details_file, database, model, sub_method, run_num)
                    else:
                        # Standard structure
                        metrics_file = method_dir / 'rate' / 'vuln_density' / 'SALLM_metrics.txt'
                        if metrics_file.exists():
                            metrics = self.parse_metrics_file(metrics_file)
                            metrics['database'] = database
                            metrics['model'] = model
                            metrics['method'] = method_name
                            metrics['run_number'] = run_num
                            metrics['method_dir'] = method_dir.name
                            self.data.append(metrics)
                            self.method_groups[method_name].append(metrics)
                        
                        # Collect details_result.xlsx
                        details_file = method_dir / 'details_result' / 'details_result.xlsx'
                        if details_file.exists():
                            self.collect_details_result(details_file, database, model, method_name, run_num)
    
    def collect_details_result(self, filepath: Path, database: str, model: str, method: str, run_num: int):
        """Collect data from details_result.xlsx files."""
        try:
            df = pd.read_excel(filepath)
            df['database'] = database
            df['model'] = model
            df['technique'] = method
            df['run_number'] = run_num
            self.details_data.append(df)
        except Exception as e:
            print(f"Warning: Could not read {filepath}: {e}")
    
    def count_decimals(self, value):
        """Count decimal places in a value."""
        try:
            d = Decimal(str(value))
            return max(0, -d.as_tuple().exponent)
        except:
            return 0
    
    def process_details_data(self):
        """Process details_result.xlsx files similar to the second script."""
        if not self.details_data:
            return None, None
        
        all_data = pd.concat(self.details_data, ignore_index=True)
        
        # Remove run numbers from technique names
        all_data["technique"] = all_data["technique"].astype(str).str.replace(r"\(\d+\)$", "", regex=True)
        
        group_cols = ["database", "model", "technique"]
        summary_rows = []
        
        for _, group in all_data.groupby(group_cols):
            numeric_cols = group.select_dtypes(include="number").columns
            decimal_map = {}
            for col in numeric_cols:
                first_val = group[col].dropna()
                if len(first_val) > 0:
                    decimal_map[col] = self.count_decimals(first_val.iloc[0])
                else:
                    decimal_map[col] = 2
            
            means = group[numeric_cols].mean()
            for col in numeric_cols:
                means[col] = round(means[col], decimal_map[col])
            
            base_row = group.iloc[0].copy()
            for col in numeric_cols:
                base_row[col] = means[col]
            
            summary_rows.append(base_row)
        
        summary_df = pd.DataFrame(summary_rows)
        summary_df = summary_df[all_data.columns]
        summary_df = summary_df.sort_values(by=["database", "model", "technique"]).reset_index(drop=True)
        
        if 'bandit_rate' in summary_df.columns:
            summary_df["bandit_rate"] = summary_df["bandit_rate"].round(2)
        if 'bandit_density' in summary_df.columns:
            summary_df["bandit_density"] = summary_df["bandit_density"].round(2)
        
        # Calculate reduction rates
        reduction_rows = []
        
        for (dataset, model), group in summary_df.groupby(["dataset", "model"]):
            direct = group[group["technique"] == "direct"]
            if direct.empty:
                continue
            
            direct = direct.iloc[0]
            
            d_bandit_rate = direct.get("bandit_rate", 0)
            d_bandit_density = direct.get("bandit_density", 0)
            d_semgrep_rate = direct.get("semgrep_rate", 0)
            d_semgrep_density = direct.get("semgrep_density", 0)
            
            for _, row in group.iterrows():
                def safe_reduction(base, value):
                    if pd.isna(base) or base == 0:
                        return 0
                    return ((base - value) / base) * 100
                
                reduction_rows.append({
                    "dataset": row["dataset"],
                    "model": row["model"],
                    "technique": row["technique"],
                    "bandit_rate(*100)": row.get("bandit_rate", 0) * 100,
                    "bandit_density(*1000)": row.get("bandit_density", 0) * 1000,
                    "semgrep_rate(*100)": row.get("semgrep_rate", 0) * 100,
                    "semgrep_density(*1000)": row.get("semgrep_density", 0) * 1000,
                    "%▼bandit_rate": round(safe_reduction(d_bandit_rate, row.get("bandit_rate", 0)), 2),
                    "%▼bandit_density": round(safe_reduction(d_bandit_density, row.get("bandit_density", 0)), 2),
                    "%▼semgrep_rate": round(safe_reduction(d_semgrep_rate, row.get("semgrep_rate", 0)), 2),
                    "%▼semgrep_density": round(safe_reduction(d_semgrep_density, row.get("semgrep_density", 0)), 2),
                })
        
        reduction_df = pd.DataFrame(reduction_rows)
        
        if not reduction_df.empty:
            reduction_df = reduction_df[
                ["dataset", "model", "technique",
                 "bandit_rate(*100)", "bandit_density(*1000)",
                 "semgrep_rate(*100)", "semgrep_density(*1000)",
                 "%▼bandit_rate", "%▼bandit_density",
                 "%▼semgrep_rate", "%▼semgrep_density"]
            ]
            reduction_df = reduction_df.sort_values(by=["dataset", "model", "technique"]).reset_index(drop=True)
        
        return summary_df, reduction_df
    
    def create_summary_report(self):
        """Create a comprehensive summary report."""
        df = pd.DataFrame(self.data)
        
        if df.empty:
            print("No data collected. Check input directory structure.")
            return
        
        # Save raw data
        df.to_csv(self.output_dir / 'raw_data.csv', index=False)
        
        # Create summary statistics grouped by method
        summary_cols = ['method', 'bandit_rate', 'bandit_density', 'semgrep_rate', 
                       'semgrep_density', 'total_tokens', 'runtime_seconds']
        
        available_cols = [col for col in summary_cols if col in df.columns]
        
        if len(available_cols) > 1:
            summary = df.groupby('method')[available_cols[1:]].agg(['mean', 'std', 'min', 'max', 'count'])
            summary.to_csv(self.output_dir / 'method_summary.csv')
            
            # Create a flattened version for easier reading
            summary_flat = df.groupby('method').agg({
                col: ['mean', 'std'] for col in available_cols[1:]
            })
            summary_flat.columns = ['_'.join(col).strip() for col in summary_flat.columns.values]
            summary_flat.to_csv(self.output_dir / 'method_summary_flat.csv')
    
    def create_comparison_tables(self):
        """Create detailed comparison tables."""
        df = pd.DataFrame(self.data)
        
        if df.empty:
            return
        
        # Vulnerability Rate Comparison
        if 'bandit_rate' in df.columns and 'semgrep_rate' in df.columns:
            rate_comparison = df.pivot_table(
                values=['bandit_rate', 'semgrep_rate'],
                index='method',
                aggfunc=['mean', 'std', 'count']
            )
            rate_comparison.to_csv(self.output_dir / 'vulnerability_rate_comparison.csv')
        
        # Vulnerability Density Comparison
        if 'bandit_density' in df.columns and 'semgrep_density' in df.columns:
            density_comparison = df.pivot_table(
                values=['bandit_density', 'semgrep_density'],
                index='method',
                aggfunc=['mean', 'std', 'count']
            )
            density_comparison.to_csv(self.output_dir / 'vulnerability_density_comparison.csv')
        
        # Token Usage Comparison
        if 'total_tokens' in df.columns:
            token_comparison = df.pivot_table(
                values=['prompt_tokens', 'completion_tokens', 'total_tokens'],
                index='method',
                aggfunc=['mean', 'std']
            )
            token_comparison.to_csv(self.output_dir / 'token_usage_comparison.csv')
        
        # Runtime Comparison
        if 'runtime_seconds' in df.columns:
            runtime_comparison = df.pivot_table(
                values='runtime_seconds',
                index='method',
                aggfunc=['mean', 'std', 'min', 'max']
            )
            runtime_comparison.to_csv(self.output_dir / 'runtime_comparison.csv')
        
        # Detailed per-run breakdown
        detailed = df[['method', 'run_number', 'bandit_rate', 'bandit_density', 
                      'semgrep_rate', 'semgrep_density', 'total_tokens', 'runtime_seconds']]
        detailed = detailed.sort_values(['method', 'run_number'])
        detailed.to_csv(self.output_dir / 'detailed_per_run.csv', index=False)
    
    def create_iteration_analysis(self):
        """Analyze methods with iterations (rci_iter, self_refine_iter, planning)."""
        df = pd.DataFrame(self.data)
        
        if df.empty:
            return
        
        # Extract iteration number for iterative methods
        df['base_method'] = df['method'].str.replace(r'_iter\d+$', '', regex=True)
        df['iteration'] = df['method'].str.extract(r'_iter(\d+)$').fillna(0).astype(int)
        
        # Filter iterative methods
        iterative_methods = df[df['iteration'] > 0]
        
        if not iterative_methods.empty:
            # Group by base method and iteration
            iter_analysis = iterative_methods.groupby(['base_method', 'iteration']).agg({
                'bandit_rate': ['mean', 'std'],
                'bandit_density': ['mean', 'std'],
                'semgrep_rate': ['mean', 'std'],
                'semgrep_density': ['mean', 'std'],
                'total_tokens': ['mean', 'std'],
                'runtime_seconds': ['mean', 'std']
            })
            iter_analysis.to_csv(self.output_dir / 'iteration_analysis.csv')
    
    def autosize_columns(self, ws):
        """Auto-size columns in Excel worksheet."""
        for column in ws.columns:
            max_length = 0
            col = column[0].column
            for cell in column:
                if cell.value:
                    max_length = max(max_length, len(str(cell.value)))
            ws.column_dimensions[get_column_letter(col)].width = max_length + 2
    
    def center_align(self, ws):
        """Center align all cells in Excel worksheet."""
        align = Alignment(horizontal="center", vertical="center")
        for row in ws.iter_rows():
            for cell in row:
                cell.alignment = align
    
    def create_plots(self):
        """Generate visualization plots."""
        df = pd.DataFrame(self.data)
        
        if df.empty:
            return
        
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        
        # 1. Vulnerability Rate Comparison (Bandit vs Semgrep) - WITH PLANNING HIGHLIGHTED
        if 'bandit_rate' in df.columns and 'semgrep_rate' in df.columns:
            fig, ax = plt.subplots(figsize=(14, 8))
            
            method_means = df.groupby('method')[['bandit_rate', 'semgrep_rate']].mean()
            
            # Create color arrays for highlighting planning
            colors_bandit = ['#FF6B6B' if method == 'planning' else '#D3D3D3' for method in method_means.index]
            colors_semgrep = ['#4ECDC4' if method == 'planning' else '#A9A9A9' for method in method_means.index]
            
            x = range(len(method_means))
            width = 0.35
            
            ax.bar([i - width/2 for i in x], method_means['bandit_rate'], width, 
                   label='Bandit', color=colors_bandit)
            ax.bar([i + width/2 for i in x], method_means['semgrep_rate'], width, 
                   label='Semgrep', color=colors_semgrep)
            
            ax.set_title('Vulnerability Rate by Method (Bandit vs Semgrep)', fontsize=16, fontweight='bold')
            ax.set_xlabel('Method', fontsize=12)
            ax.set_ylabel('Vulnerability Rate', fontsize=12)
            ax.set_xticks(x)
            ax.set_xticklabels(method_means.index, rotation=45, ha='right')
            ax.legend(['Bandit', 'Semgrep'], fontsize=10)
            ax.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.output_dir / 'vulnerability_rate_comparison.png', dpi=300)
            plt.close()
        
        # 2. Vulnerability Density Comparison - WITH PLANNING HIGHLIGHTED
        if 'bandit_density' in df.columns and 'semgrep_density' in df.columns:
            fig, ax = plt.subplots(figsize=(14, 8))
            
            method_means = df.groupby('method')[['bandit_density', 'semgrep_density']].mean()
            
            # Create color arrays for highlighting planning
            colors_bandit = ['#95E1D3' if method == 'planning' else '#D3D3D3' for method in method_means.index]
            colors_semgrep = ['#F38181' if method == 'planning' else '#A9A9A9' for method in method_means.index]
            
            x = range(len(method_means))
            width = 0.35
            
            ax.bar([i - width/2 for i in x], method_means['bandit_density'], width, 
                   label='Bandit', color=colors_bandit)
            ax.bar([i + width/2 for i in x], method_means['semgrep_density'], width, 
                   label='Semgrep', color=colors_semgrep)
            
            ax.set_title('Vulnerability Density by Method (Bandit vs Semgrep)', fontsize=16, fontweight='bold')
            ax.set_xlabel('Method', fontsize=12)
            ax.set_ylabel('Vulnerability Density (issues/LOC)', fontsize=12)
            ax.set_xticks(x)
            ax.set_xticklabels(method_means.index, rotation=45, ha='right')
            ax.legend(['Bandit', 'Semgrep'], fontsize=10)
            ax.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.output_dir / 'vulnerability_density_comparison.png', dpi=300)
            plt.close()
        
        # 3. Token Usage Comparison
        if 'total_tokens' in df.columns:
            fig, ax = plt.subplots(figsize=(14, 8))
            
            token_means = df.groupby('method')['total_tokens'].mean().sort_values()
            token_means.plot(kind='barh', ax=ax, color='#6C5CE7')
            
            ax.set_title('Average Token Usage by Method', fontsize=16, fontweight='bold')
            ax.set_xlabel('Total Tokens', fontsize=12)
            ax.set_ylabel('Method', fontsize=12)
            ax.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.output_dir / 'token_usage_comparison.png', dpi=300)
            plt.close()
        
        # 4. Runtime Comparison
        if 'runtime_seconds' in df.columns:
            fig, ax = plt.subplots(figsize=(14, 8))
            
            runtime_means = df.groupby('method')['runtime_seconds'].mean().sort_values()
            runtime_means.plot(kind='barh', ax=ax, color='#FD79A8')
            
            ax.set_title('Average Runtime by Method', fontsize=16, fontweight='bold')
            ax.set_xlabel('Runtime (seconds)', fontsize=12)
            ax.set_ylabel('Method', fontsize=12)
            ax.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.output_dir / 'runtime_comparison.png', dpi=300)
            plt.close()
        
        # 5. Iteration Effect Analysis
        df['base_method'] = df['method'].str.replace(r'_iter\d+$', '', regex=True)
        df['iteration'] = df['method'].str.extract(r'_iter(\d+)$').fillna(0).astype(int)
        
        iterative_df = df[df['iteration'] > 0]
        
        if not iterative_df.empty and 'bandit_rate' in iterative_df.columns:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            for idx, (metric, title) in enumerate([
                ('bandit_rate', 'Bandit Vulnerability Rate'),
                ('semgrep_rate', 'Semgrep Vulnerability Rate'),
                ('bandit_density', 'Bandit Vulnerability Density'),
                ('semgrep_density', 'Semgrep Vulnerability Density')
            ]):
                if metric in iterative_df.columns:
                    ax = axes[idx // 2, idx % 2]
                    
                    for method in iterative_df['base_method'].unique():
                        method_data = iterative_df[iterative_df['base_method'] == method]
                        iter_means = method_data.groupby('iteration')[metric].mean()
                        ax.plot(iter_means.index, iter_means.values, marker='o', label=method, linewidth=2)
                    
                    ax.set_title(f'{title} vs Iterations', fontsize=12, fontweight='bold')
                    ax.set_xlabel('Iteration', fontsize=10)
                    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=10)
                    ax.legend(fontsize=8)
                    ax.grid(alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'iteration_effect_analysis.png', dpi=300)
            plt.close()
        
        # 6. Box Plot for Vulnerability Rate Distribution
        if 'semgrep_rate' in df.columns:
            fig, ax = plt.subplots(figsize=(14, 8))
            sns.boxplot(data=df, x='method', y='semgrep_rate', palette='Set3', ax=ax)
            ax.set_title('Semgrep Vulnerability Rate Distribution', fontsize=16, fontweight='bold')
            ax.set_xlabel('Method', fontsize=12)
            ax.set_ylabel('Semgrep Vulnerability Rate', fontsize=12)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(self.output_dir / 'semgrep_rate_distribution.png', dpi=300)
            plt.close()
        
        # 7. Box Plot for Density Distribution (new requirement)
        if 'semgrep_density' in df.columns:
            fig, ax = plt.subplots(figsize=(14, 8))
            sns.boxplot(data=df, x='method', y='semgrep_density', palette='coolwarm', ax=ax)
            ax.set_title('Semgrep Vulnerability Density Distribution', fontsize=16, fontweight='bold')
            ax.set_xlabel('Method', fontsize=12)
            ax.set_ylabel('Vulnerability Density (Semgrep)', fontsize=12)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(self.output_dir / 'semgrep_density_distribution.png', dpi=300)
            plt.close()

        # 8. If reduction data is available, plot density reduction
        summary_df, reduction_df = self.process_details_data()
        if reduction_df is not None and not reduction_df.empty:
            fig, ax = plt.subplots(figsize=(14, 8))
            sns.barplot(data=reduction_df, x='technique', y='%▼semgrep_density', palette='Purples', ax=ax)
            ax.set_title('Semgrep Density Reduction Rate by Technique', fontsize=16, fontweight='bold')
            ax.set_xlabel('Technique', fontsize=12)
            ax.set_ylabel('% Reduction in Semgrep Density', fontsize=12)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(self.output_dir / 'semgrep_density_reduction.png', dpi=300)
            plt.close()
    
    def create_excel_report(self):
        """Compile results into a comprehensive Excel workbook."""
        excel_path = self.output_dir / 'comprehensive_report.xlsx'
        with pd.ExcelWriter(excel_path) as writer:
            raw_df = pd.DataFrame(self.data)
            if not raw_df.empty:
                raw_df.to_excel(writer, index=False, sheet_name='Raw Data')
            
            # Method Summary
            summary_path = self.output_dir / 'method_summary_flat.csv'
            if summary_path.exists():
                summary_df = pd.read_csv(summary_path)
                summary_df.to_excel(writer, index=False, sheet_name='Method Summary')
            
            # Per Run Details
            detailed_path = self.output_dir / 'detailed_per_run.csv'
            if detailed_path.exists():
                detailed_df = pd.read_csv(detailed_path)
                detailed_df.to_excel(writer, index=False, sheet_name='Per Run Details')
            
            # Iteration Analysis
            iter_path = self.output_dir / 'iteration_analysis.csv'
            if iter_path.exists():
                iter_df = pd.read_csv(iter_path)
                iter_df.to_excel(writer, index=False, sheet_name='Iteration Analysis')
            
            # Rankings (placeholder)
            if not raw_df.empty:
                rank_df = raw_df.groupby('method')[['semgrep_rate', 'semgrep_density']].mean().sort_values(
                    by='semgrep_rate', ascending=True
                )
                rank_df.to_excel(writer, sheet_name='Rankings')
            
            # Details Summary if available
            summary_df, reduction_df = self.process_details_data()
            if summary_df is not None:
                summary_df.to_excel(writer, index=False, sheet_name='Details Summary')
            if reduction_df is not None:
                reduction_df.to_excel(writer, index=False, sheet_name='Reduction Rates')
        
        # Post-process Excel for formatting
        wb = load_workbook(excel_path)
        for ws in wb.worksheets:
            self.autosize_columns(ws)
            self.center_align(ws)
        wb.save(excel_path)
    
    def generate_report(self):
        """Run all report steps."""
        self.collect_data()
        self.create_summary_report()
        self.create_comparison_tables()
        self.create_iteration_analysis()
        self.create_plots()
        self.create_excel_report()
        print(f"✅ Report generation completed. Outputs saved in {self.output_dir}")
        

def main():
    parser = argparse.ArgumentParser(description="Consolidate security metrics and generate reports.")
    parser.add_argument("--input_dir", required=True, help="Path to input directory containing results.")
    parser.add_argument("--output_dir", required=True, help="Path to output directory for generated reports.")
    args = parser.parse_args()

    consolidator = SecurityMetricsConsolidator(args.input_dir, args.output_dir)
    consolidator.generate_report()


if __name__ == "__main__":
    main()

# python .\scripts\results.py --input_dir .\outputs\ --output_dir .\reports\ 