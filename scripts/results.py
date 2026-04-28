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

class SecurityMetricsConsolidator:
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.data = []
        self.method_groups = defaultdict(list)
        
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
            rate_match = re.search(r'Vulnerability Rate.*?=\s+([\d.]+)', bandit_text)
            density_match = re.search(r'Vulnerability Density.*?=\s+([\d.]+)', bandit_text)
            
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
            rate_match = re.search(r'Vulnerability Rate.*?=\s+([\d.]+)', semgrep_text)
            density_match = re.search(r'Vulnerability Density.*?=\s+([\d.]+)', semgrep_text)
            
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
        # Handle cases like 'direct(2)', 'rci_iter1', 'one_shot'
        match = re.match(r'([a-z_]+)(?:\((\d+)\))?', method_dir_name)
        if match:
            method = match.group(1)
            run_num = int(match.group(2)) if match.group(2) else 1
            return method, run_num
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
        df['base_method'] = df['method'].str.replace(r'_iter\d+', '', regex=True)
        df['iteration'] = df['method'].str.extract(r'_iter(\d+)').fillna(0).astype(int)
        
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
    
    def create_plots(self):
        """Generate visualization plots."""
        df = pd.DataFrame(self.data)
        
        if df.empty:
            return
        
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        
        # 1. Vulnerability Rate Comparison (Bandit vs Semgrep)
        if 'bandit_rate' in df.columns and 'semgrep_rate' in df.columns:
            fig, ax = plt.subplots(figsize=(14, 8))
            
            method_means = df.groupby('method')[['bandit_rate', 'semgrep_rate']].mean()
            method_means.plot(kind='bar', ax=ax, color=['#FF6B6B', '#4ECDC4'])
            
            ax.set_title('Vulnerability Rate by Method (Bandit vs Semgrep)', fontsize=16, fontweight='bold')
            ax.set_xlabel('Method', fontsize=12)
            ax.set_ylabel('Vulnerability Rate', fontsize=12)
            ax.legend(['Bandit', 'Semgrep'], fontsize=10)
            ax.grid(axis='y', alpha=0.3)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(self.output_dir / 'vulnerability_rate_comparison.png', dpi=300)
            plt.close()
        
        # 2. Vulnerability Density Comparison
        if 'bandit_density' in df.columns and 'semgrep_density' in df.columns:
            fig, ax = plt.subplots(figsize=(14, 8))
            
            method_means = df.groupby('method')[['bandit_density', 'semgrep_density']].mean()
            method_means.plot(kind='bar', ax=ax, color=['#95E1D3', '#F38181'])
            
            ax.set_title('Vulnerability Density by Method (Bandit vs Semgrep)', fontsize=16, fontweight='bold')
            ax.set_xlabel('Method', fontsize=12)
            ax.set_ylabel('Vulnerability Density (issues/LOC)', fontsize=12)
            ax.legend(['Bandit', 'Semgrep'], fontsize=10)
            ax.grid(axis='y', alpha=0.3)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(self.output_dir / 'vulnerability_density_comparison.png', dpi=300)
            plt.close()
        
        # 3. Combined Metrics Heatmap
        if all(col in df.columns for col in ['bandit_rate', 'semgrep_rate', 'bandit_density', 'semgrep_density']):
            fig, ax = plt.subplots(figsize=(12, 10))
            
            heatmap_data = df.groupby('method')[['bandit_rate', 'semgrep_rate', 
                                                  'bandit_density', 'semgrep_density']].mean()
            
            sns.heatmap(heatmap_data.T, annot=True, fmt='.4f', cmap='YlOrRd', 
                       ax=ax, cbar_kws={'label': 'Value'})
            
            ax.set_title('Security Metrics Heatmap by Method', fontsize=16, fontweight='bold')
            ax.set_xlabel('Method', fontsize=12)
            ax.set_ylabel('Metric', fontsize=12)
            plt.tight_layout()
            plt.savefig(self.output_dir / 'metrics_heatmap.png', dpi=300)
            plt.close()
        
        # 4. Token Usage Comparison
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
        
        # 5. Runtime Comparison
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
        
        # 6. Iteration Effect Analysis
        df['base_method'] = df['method'].str.replace(r'_iter\d+', '', regex=True)
        df['iteration'] = df['method'].str.extract(r'_iter(\d+)').fillna(0).astype(int)
        
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
                    ax.legend()
                    ax.grid(alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'iteration_effect_analysis.png', dpi=300)
            plt.close()
        
        # 7. Box plots for variability analysis
        if 'bandit_rate' in df.columns:
            fig, axes = plt.subplots(2, 1, figsize=(14, 12))
            
            df_sorted = df.sort_values('method')
            
            sns.boxplot(data=df_sorted, x='method', y='bandit_rate', ax=axes[0], palette='Set2')
            axes[0].set_title('Bandit Vulnerability Rate Distribution by Method', fontsize=14, fontweight='bold')
            axes[0].set_xlabel('Method', fontsize=11)
            axes[0].set_ylabel('Vulnerability Rate', fontsize=11)
            axes[0].tick_params(axis='x', rotation=45)
            axes[0].grid(axis='y', alpha=0.3)
            
            sns.boxplot(data=df_sorted, x='method', y='semgrep_rate', ax=axes[1], palette='Set3')
            axes[1].set_title('Semgrep Vulnerability Rate Distribution by Method', fontsize=14, fontweight='bold')
            axes[1].set_xlabel('Method', fontsize=11)
            axes[1].set_ylabel('Vulnerability Rate', fontsize=11)
            axes[1].tick_params(axis='x', rotation=45)
            axes[1].grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'vulnerability_rate_distribution.png', dpi=300)
            plt.close()
    
    def create_excel_report(self):
        """Create a comprehensive Excel report with multiple sheets."""
        df = pd.DataFrame(self.data)
        
        if df.empty:
            return
        
        excel_path = self.output_dir / 'comprehensive_report.xlsx'
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Sheet 1: Raw Data
            df.to_excel(writer, sheet_name='Raw Data', index=False)
            
            # Sheet 2: Method Summary
            summary = df.groupby('method').agg({
                'bandit_rate': ['mean', 'std', 'min', 'max'],
                'bandit_density': ['mean', 'std', 'min', 'max'],
                'semgrep_rate': ['mean', 'std', 'min', 'max'],
                'semgrep_density': ['mean', 'std', 'min', 'max'],
                'total_tokens': ['mean', 'std'],
                'runtime_seconds': ['mean', 'std']
            })
            summary.to_excel(writer, sheet_name='Method Summary')
            
            # Sheet 3: Detailed Per Run
            detailed = df[['method', 'run_number', 'bandit_rate', 'bandit_density', 
                          'semgrep_rate', 'semgrep_density', 'total_tokens', 'runtime_seconds']]
            detailed = detailed.sort_values(['method', 'run_number'])
            detailed.to_excel(writer, sheet_name='Per Run Details', index=False)
            
            # Sheet 4: Iteration Analysis
            df['base_method'] = df['method'].str.replace(r'_iter\d+', '', regex=True)
            df['iteration'] = df['method'].str.extract(r'_iter(\d+)').fillna(0).astype(int)
            iterative_df = df[df['iteration'] > 0]
            
            if not iterative_df.empty:
                iter_analysis = iterative_df.groupby(['base_method', 'iteration']).agg({
                    'bandit_rate': ['mean', 'std'],
                    'semgrep_rate': ['mean', 'std'],
                    'bandit_density': ['mean', 'std'],
                    'semgrep_density': ['mean', 'std']
                })
                iter_analysis.to_excel(writer, sheet_name='Iteration Analysis')
            
            # Sheet 5: Rankings
            rankings = pd.DataFrame({
                'Method': df.groupby('method')['bandit_rate'].mean().sort_values().index,
                'Avg Bandit Rate': df.groupby('method')['bandit_rate'].mean().sort_values().values,
                'Bandit Rank': range(1, len(df['method'].unique()) + 1)
            })
            rankings.to_excel(writer, sheet_name='Rankings', index=False)
        
        print(f"Excel report created: {excel_path}")
    
    def generate_report(self):
        """Main method to generate all reports."""
        print("Collecting data...")
        self.collect_data()
        
        if not self.data:
            print("No data found. Please check the input directory structure.")
            return
        
        print(f"Collected {len(self.data)} data points from {len(self.method_groups)} methods.")
        
        print("Creating summary report...")
        self.create_summary_report()
        
        print("Creating comparison tables...")
        self.create_comparison_tables()
        
        print("Creating iteration analysis...")
        self.create_iteration_analysis()
        
        print("Generating plots...")
        self.create_plots()
        
        print("Creating Excel report...")
        self.create_excel_report()
        
        print(f"\nAll reports generated successfully in: {self.output_dir}")
        print("\nGenerated files:")
        for file in sorted(self.output_dir.iterdir()):
            print(f"  - {file.name}")


def main():
    parser = argparse.ArgumentParser(
        description='Consolidate LLM code generation security analysis results'
    )
    parser.add_argument(
        'input_dir',
        type=str,
        help='Input directory containing the results'
    )
    parser.add_argument(
        'output_dir',
        type=str,
        help='Output directory for consolidated results'
    )
    
    args = parser.parse_args()
    
    consolidator = SecurityMetricsConsolidator(args.input_dir, args.output_dir)
    consolidator.generate_report()


if __name__ == '__main__':
    main()
