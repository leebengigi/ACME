#!/usr/bin/env python3
"""
Classification Results Visualizer
Generate tables and plots to summarize classification performance
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time
from collections import defaultdict, Counter
from tabulate import tabulate

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.enums import RequestType
from src.services.classification_service import ClassificationService

class ClassificationVisualizer:
    """Generate visual summaries of classification performance"""
    
    def __init__(self):
        self.classifier = ClassificationService()
        self.results_data = []
        
        # Set up matplotlib style
        plt.style.use('default')
        sns.set_palette("husl")
        
    def run_performance_test(self, sample_size: int = 100) -> pd.DataFrame:
        """Run classification tests and collect results"""
        
        print(f"🧪 Running performance test with {sample_size} samples...")
        
        # Prepare test cases
        test_cases = self._prepare_test_cases(sample_size)
        
        results = []
        layer_results = defaultdict(list)
        
        for i, (text, true_type) in enumerate(test_cases):
            if i % 20 == 0:
                print(f"   Testing {i+1}/{len(test_cases)}...")
            
            # Test overall system
            start_time = time.time()
            pred_type, confidence = self.classifier.classify_request(text)
            overall_time = time.time() - start_time
            
            # Determine which layer was likely used
            if hasattr(self.classifier, 'has_llm') and self.classifier.has_llm and confidence > 0.85:
                layer_used = 'LLM'
            elif confidence > 0.8:
                layer_used = 'Enhanced_Patterns'
            elif confidence > 0.6:
                layer_used = 'ML_Ensemble'
            else:
                layer_used = 'Keyword_Fallback'
            
            # Record result
            result = {
                'text': text[:100],  # Truncate for display
                'true_type': true_type.value,
                'predicted_type': pred_type.value,
                'confidence': confidence,
                'processing_time_ms': overall_time * 1000,
                'layer_used': layer_used,
                'correct': pred_type == true_type,
                'text_length': len(text),
                'word_count': len(text.split())
            }
            results.append(result)
            
            # Test individual layers for comparison
            self._test_individual_layers(text, true_type, layer_results)
        
        self.results_data = results
        return pd.DataFrame(results)
    
    def _prepare_test_cases(self, sample_size: int):
        """Prepare test cases from CSV and custom examples"""
        
        test_cases = []
        
        # Custom test cases (high quality)
        custom_cases = [
            ("Onboard vendor Microsoft for Office 365 integration", RequestType.VENDOR_APPROVAL),
            ("Approve contractor for security audit services", RequestType.VENDOR_APPROVAL),
            ("Need approval for third-party analytics platform", RequestType.VENDOR_APPROVAL),
            ("Request vendor approval for AWS partnership", RequestType.VENDOR_APPROVAL),
            
            ("Need admin access to production database", RequestType.PERMISSION_CHANGE),
            ("Request elevated permissions for system maintenance", RequestType.PERMISSION_CHANGE),
            ("Temporary sudo access for server patching", RequestType.PERMISSION_CHANGE),
            ("Root access needed for security updates", RequestType.PERMISSION_CHANGE),
            
            ("Open port 443 to external API endpoint", RequestType.NETWORK_ACCESS),
            ("Need firewall rule for SSH access to 10.0.1.5", RequestType.NETWORK_ACCESS),
            ("Allow connection from office to production VPN", RequestType.NETWORK_ACCESS),
            ("Whitelist IP 192.168.1.100 for database access", RequestType.NETWORK_ACCESS),
            
            ("Need access to customer database for analysis", RequestType.DATA_EXPORT),
            ("Export user data for GDPR compliance report", RequestType.DATA_EXPORT),
            ("Query production logs for debugging issue", RequestType.DATA_EXPORT),
            ("Read access to sales data for quarterly report", RequestType.DATA_EXPORT),
            
            ("SSH access to AWS EC2 for deployment", RequestType.CLOUD_RESOURCE_ACCESS),
            ("Need login to Azure staging environment", RequestType.CLOUD_RESOURCE_ACCESS),
            ("Access to GCP servers for monitoring", RequestType.CLOUD_RESOURCE_ACCESS),
            ("Connect to development server for testing", RequestType.CLOUD_RESOURCE_ACCESS),
        ]
        
        test_cases.extend(custom_cases)
        
        # Load CSV data if available
        try:
            df = pd.read_csv(r"data\acme_security_tickets.csv")
            
            # Map CSV labels
            label_mapping = {
                'vendor approval': RequestType.VENDOR_APPROVAL,
                'permission change': RequestType.PERMISSION_CHANGE,
                'permission escalation': RequestType.PERMISSION_CHANGE,
                'network access': RequestType.NETWORK_ACCESS,
                'firewall change': RequestType.FIREWALL_CHANGE,
                'devtool install': RequestType.DEVTOOL_INSTALL,
                'data export': RequestType.DATA_EXPORT,
                'data access': RequestType.DATA_EXPORT,
                'cloud resource access': RequestType.CLOUD_RESOURCE_ACCESS,
                'system access': RequestType.CLOUD_RESOURCE_ACCESS,
            }
            
            df['request_type_clean'] = df['request_type'].str.lower().str.strip()
            df['request_type_enum'] = df['request_type_clean'].map(label_mapping)
            df = df.dropna(subset=['request_summary', 'request_type_enum'])
            
            # Add CSV samples
            csv_sample_size = min(sample_size - len(custom_cases), len(df))
            if csv_sample_size > 0:
                csv_sample = df.sample(n=csv_sample_size, random_state=42)
                for _, row in csv_sample.iterrows():
                    test_cases.append((row['request_summary'], row['request_type_enum']))
            
        except FileNotFoundError:
            print("   No CSV file found, using custom cases only")
        
        return test_cases[:sample_size]
    
    def _test_individual_layers(self, text: str, true_type: RequestType, layer_results: dict):
        """Test individual layers for comparison"""
        
        # Test Enhanced Patterns
        try:
            pattern_result = self.classifier._classify_with_enhanced_patterns(text)
            if pattern_result:
                layer_results['Enhanced_Patterns'].append({
                    'correct': pattern_result[0] == true_type,
                    'confidence': pattern_result[1]
                })
        except:
            pass
        
        # Test Keyword Fallback
        try:
            keyword_result = self.classifier._simple_keyword_classification(text)
            layer_results['Keyword_Fallback'].append({
                'correct': keyword_result[0] == true_type,
                'confidence': keyword_result[1]
            })
        except:
            pass
        
        # Test LLM if available
        if hasattr(self.classifier, 'has_llm') and self.classifier.has_llm:
            try:
                llm_result = self.classifier.llm_classifier.classify_request(text)
                layer_results['LLM'].append({
                    'correct': llm_result[0] == true_type,
                    'confidence': llm_result[1]
                })
            except:
                pass
    
    def generate_summary_table(self, df: pd.DataFrame) -> str:
        """Generate comprehensive summary table"""
        
        print("\n📊 GENERATING SUMMARY TABLES...")
        
        # Overall Performance Summary
        total_tests = len(df)
        overall_accuracy = df['correct'].mean()
        avg_confidence = df['confidence'].mean()
        avg_time = df['processing_time_ms'].mean()
        
        summary_data = []
        
        # Performance by Layer
        layer_stats = df.groupby('layer_used').agg({
            'correct': ['count', 'sum', 'mean'],
            'confidence': 'mean',
            'processing_time_ms': 'mean'
        }).round(3)
        
        for layer in layer_stats.index:
            stats = layer_stats.loc[layer]
            summary_data.append([
                layer,
                int(stats[('correct', 'count')]),  # Total predictions
                int(stats[('correct', 'sum')]),    # Correct predictions
                f"{stats[('correct', 'mean')]:.1%}",  # Accuracy
                f"{stats[('confidence', 'mean')]:.2f}",  # Avg confidence
                f"{stats[('processing_time_ms', 'mean')]:.1f}ms"  # Avg time
            ])
        
        # Performance by Request Type
        type_stats = df.groupby('true_type').agg({
            'correct': ['count', 'mean'],
            'confidence': 'mean'
        }).round(3)
        
        type_summary = []
        for req_type in type_stats.index:
            stats = type_stats.loc[req_type]
            type_summary.append([
                req_type,
                int(stats[('correct', 'count')]),
                f"{stats[('correct', 'mean')]:.1%}",
                f"{stats[('confidence', 'mean')]:.2f}"
            ])
        
        # Generate tables
        table1 = tabulate(
            summary_data,
            headers=['Layer', 'Total', 'Correct', 'Accuracy', 'Avg Confidence', 'Avg Time'],
            tablefmt='grid',
            floatfmt='.2f'
        )
        
        table2 = tabulate(
            type_summary,
            headers=['Request Type', 'Tests', 'Accuracy', 'Avg Confidence'],
            tablefmt='grid',
            floatfmt='.2f'
        )
        
        # Confusion Matrix
        confusion_data = []
        confusion_matrix = pd.crosstab(df['true_type'], df['predicted_type'], margins=True)
        
        # Print results
        output = f"""
🎯 CLASSIFICATION PERFORMANCE SUMMARY
{'='*60}

📈 OVERALL STATISTICS:
   Total Tests: {total_tests}
   Overall Accuracy: {overall_accuracy:.1%}
   Average Confidence: {avg_confidence:.2f}
   Average Processing Time: {avg_time:.1f}ms

📊 PERFORMANCE BY LAYER:
{table1}

🎯 PERFORMANCE BY REQUEST TYPE:
{table2}

🔄 CONFUSION MATRIX:
{confusion_matrix.to_string()}
"""
        
        return output
    
    def create_performance_plots(self, df: pd.DataFrame):
        """Create comprehensive performance visualization plots"""
        
        print("\n📈 GENERATING PERFORMANCE PLOTS...")
        
        # Create subplot layout
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Classification Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. Accuracy by Layer
        layer_accuracy = df.groupby('layer_used')['correct'].mean().sort_values(ascending=False)
        
        axes[0,0].bar(range(len(layer_accuracy)), layer_accuracy.values, 
                     color=['#2E8B57', '#4169E1', '#FF6347', '#FFD700'])
        axes[0,0].set_title('Accuracy by Classification Layer', fontweight='bold')
        axes[0,0].set_ylabel('Accuracy')
        axes[0,0].set_xticks(range(len(layer_accuracy)))
        axes[0,0].set_xticklabels(layer_accuracy.index, rotation=45, ha='right')
        axes[0,0].set_ylim(0, 1)
        
        # Add value labels on bars
        for i, v in enumerate(layer_accuracy.values):
            axes[0,0].text(i, v + 0.01, f'{v:.1%}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Confidence vs Accuracy Scatter
        axes[0,1].scatter(df['confidence'], df['correct'].astype(int), 
                         alpha=0.6, c=df['correct'].astype(int), cmap='RdYlGn')
        axes[0,1].set_title('Confidence vs Accuracy', fontweight='bold')
        axes[0,1].set_xlabel('Confidence Score')
        axes[0,1].set_ylabel('Correct (1) / Incorrect (0)')
        axes[0,1].set_ylim(-0.1, 1.1)
        
        # Add trend line
        z = np.polyfit(df['confidence'], df['correct'].astype(int), 1)
        p = np.poly1d(z)
        axes[0,1].plot(df['confidence'].sort_values(), p(df['confidence'].sort_values()), 
                      "r--", alpha=0.8, linewidth=2)
        
        # 3. Processing Time by Layer
        layer_time = df.groupby('layer_used')['processing_time_ms'].mean().sort_values()
        
        axes[0,2].barh(range(len(layer_time)), layer_time.values, 
                      color=['#FFA500', '#20B2AA', '#9370DB', '#DC143C'])
        axes[0,2].set_title('Average Processing Time by Layer', fontweight='bold')
        axes[0,2].set_xlabel('Processing Time (ms)')
        axes[0,2].set_yticks(range(len(layer_time)))
        axes[0,2].set_yticklabels(layer_time.index)
        
        # Add value labels
        for i, v in enumerate(layer_time.values):
            axes[0,2].text(v + 0.5, i, f'{v:.1f}ms', va='center', fontweight='bold')
        
        # 4. Accuracy by Request Type
        type_accuracy = df.groupby('true_type')['correct'].mean().sort_values(ascending=False)
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(type_accuracy)))
        bars = axes[1,0].bar(range(len(type_accuracy)), type_accuracy.values, color=colors)
        axes[1,0].set_title('Accuracy by Request Type', fontweight='bold')
        axes[1,0].set_ylabel('Accuracy')
        axes[1,0].set_xticks(range(len(type_accuracy)))
        axes[1,0].set_xticklabels([t.replace('_', '\n') for t in type_accuracy.index], 
                                 rotation=0, ha='center', fontsize=9)
        axes[1,0].set_ylim(0, 1)
        
        # Add value labels
        for i, v in enumerate(type_accuracy.values):
            axes[1,0].text(i, v + 0.01, f'{v:.1%}', ha='center', va='bottom', fontweight='bold')
        
        # 5. Layer Usage Distribution
        layer_counts = df['layer_used'].value_counts()
        
        axes[1,1].pie(layer_counts.values, labels=layer_counts.index, autopct='%1.1f%%',
                     colors=['#FF9999', '#66B2FF', '#99FF99', '#FFCC99'])
        axes[1,1].set_title('Layer Usage Distribution', fontweight='bold')
        
        # 6. Confidence Distribution by Correctness
        correct_conf = df[df['correct'] == True]['confidence']
        incorrect_conf = df[df['correct'] == False]['confidence']
        
        axes[1,2].hist(correct_conf, bins=20, alpha=0.7, label='Correct', color='green', density=True)
        axes[1,2].hist(incorrect_conf, bins=20, alpha=0.7, label='Incorrect', color='red', density=True)
        axes[1,2].set_title('Confidence Distribution', fontweight='bold')
        axes[1,2].set_xlabel('Confidence Score')
        axes[1,2].set_ylabel('Density')
        axes[1,2].legend()
        
        plt.tight_layout()
        plt.savefig('classification_performance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Performance plots saved as 'classification_performance.png'")
    
    def create_confusion_matrix_heatmap(self, df: pd.DataFrame):
        """Create detailed confusion matrix heatmap"""
        
        print("\n🔥 GENERATING CONFUSION MATRIX...")
        
        # Create confusion matrix
        confusion_matrix = pd.crosstab(df['true_type'], df['predicted_type'])
        
        # Calculate percentages
        confusion_pct = confusion_matrix.div(confusion_matrix.sum(axis=1), axis=0) * 100
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        
        # Plot with annotations
        sns.heatmap(confusion_pct, annot=True, fmt='.1f', cmap='Blues', 
                   cbar_kws={'label': 'Percentage (%)'})
        
        plt.title('Classification Confusion Matrix (%)', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Predicted Type', fontsize=12, fontweight='bold')
        plt.ylabel('True Type', fontsize=12, fontweight='bold')
        
        # Rotate labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Confusion matrix saved as 'confusion_matrix.png'")
        
        return confusion_matrix
    
    def generate_layer_comparison_table(self):
        """Generate detailed layer comparison"""
        
        print("\n🔍 TESTING INDIVIDUAL LAYERS...")
        
        # Test each layer individually on sample cases
        test_cases = [
            ("Onboard vendor Microsoft for analytics", RequestType.VENDOR_APPROVAL),
            ("Need admin access to production server", RequestType.PERMISSION_CHANGE),
            ("Open port 443 for API access", RequestType.NETWORK_ACCESS),
            ("Export customer data for report", RequestType.DATA_EXPORT),
            ("SSH access to AWS instance", RequestType.CLOUD_RESOURCE_ACCESS),
        ]
        
        layer_results = {
            'Enhanced_Patterns': [],
            'Keyword_Fallback': [],
            'LLM': []
        }
        
        for text, true_type in test_cases:
            # Test Enhanced Patterns
            try:
                pattern_result = self.classifier._classify_with_enhanced_patterns(text)
                if pattern_result:
                    layer_results['Enhanced_Patterns'].append({
                        'correct': pattern_result[0] == true_type,
                        'confidence': pattern_result[1]
                    })
            except:
                layer_results['Enhanced_Patterns'].append({'correct': False, 'confidence': 0.0})
            
            # Test Keyword Fallback
            try:
                keyword_result = self.classifier._simple_keyword_classification(text)
                layer_results['Keyword_Fallback'].append({
                    'correct': keyword_result[0] == true_type,
                    'confidence': keyword_result[1]
                })
            except:
                layer_results['Keyword_Fallback'].append({'correct': False, 'confidence': 0.0})
            
            # Test LLM
            if hasattr(self.classifier, 'has_llm') and self.classifier.has_llm:
                try:
                    llm_result = self.classifier.llm_classifier.classify_request(text)
                    layer_results['LLM'].append({
                        'correct': llm_result[0] == true_type,
                        'confidence': llm_result[1]
                    })
                except:
                    layer_results['LLM'].append({'correct': False, 'confidence': 0.0})
        
        # Create comparison table
        comparison_data = []
        for layer_name, results in layer_results.items():
            if results:
                accuracy = np.mean([r['correct'] for r in results])
                avg_conf = np.mean([r['confidence'] for r in results])
                comparison_data.append([
                    layer_name,
                    len(results),
                    f"{accuracy:.1%}",
                    f"{avg_conf:.2f}"
                ])
        
        table = tabulate(
            comparison_data,
            headers=['Layer', 'Tests', 'Accuracy', 'Avg Confidence'],
            tablefmt='grid'
        )
        
        print(f"\n🔀 LAYER COMPARISON:")
        print(table)
    
    def run_complete_analysis(self, sample_size: int = 100):
        """Run complete performance analysis with visualizations"""
        
        print("🚀 STARTING COMPLETE CLASSIFICATION ANALYSIS")
        print("=" * 60)
        
        # Run performance test
        df = self.run_performance_test(sample_size)
        
        # Generate summary table
        summary = self.generate_summary_table(df)
        print(summary)
        
        # Save summary to file
        with open('classification_analysis.txt', 'w', encoding='utf-8') as f:
            f.write(summary)
        print("\n💾 Summary saved to 'classification_analysis.txt'")
        
        # Generate plots
        self.create_performance_plots(df)
        
        # Generate confusion matrix
        confusion_matrix = self.create_confusion_matrix_heatmap(df)
        
        # Layer comparison
        self.generate_layer_comparison_table()
        
        # Save detailed results
        df.to_csv('classification_results.csv', index=False)
        print("\n💾 Detailed results saved to 'classification_results.csv'")
        
        print("\n✅ ANALYSIS COMPLETE!")
        print("Generated files:")
        print("   - classification_analysis.txt")
        print("   - classification_performance.png")
        print("   - confusion_matrix.png")
        print("   - classification_results.csv")

def main():
    """Main function"""
    
    classifier = ClassificationService()
    text = "Open port 3306 from 10.81.202.0/24 to RDS cluster rds-acme-prod"
    pred_type, confidence = classifier.classify_request(text)
    print(f"Predicted type: {pred_type}, Confidence: {confidence}")

    visualizer = ClassificationVisualizer()
    
    print("📊 CLASSIFICATION PERFORMANCE VISUALIZER")
    print("=" * 50)
    
    sample_size = input("Enter sample size for testing (default 100): ").strip()
    sample_size = int(sample_size) if sample_size.isdigit() else 100
    
    visualizer.run_complete_analysis(sample_size)

if __name__ == "__main__":
    main()