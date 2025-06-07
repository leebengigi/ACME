#!/usr/bin/env python3
"""
Comprehensive Model Performance Tester
Tests both Risk Assessment and Decision Engine models with detailed analysis
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
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.enums import RequestType, Outcome
from src.services.classification_service import ClassificationService
from src.services.adaptive_bot_system import AdaptiveBotSystem
from src.services.data_service import DataService
from src.models.data_models import SecurityRequest

class ModelPerformanceTester:
    """Comprehensive tester for all AI models in the system"""
    
    def __init__(self):
        self.setup_services()
        self.test_results = {
            'classification': [],
            'risk_assessment': [],
            'decision_engine': [],
            'end_to_end': []
        }
        
        # Set up visualization style
        plt.style.use('default')
        sns.set_palette("husl")
        
    def setup_services(self):
        """Initialize all services for testing"""
        print("🔧 Initializing services for testing...")
        
        try:
            # Initialize data service and load data
            self.data_service = DataService(r"data\acme_security_tickets.csv")
            self.historical_data = self.data_service.load_and_normalize_data()
            
            # Initialize classification service
            self.classification_service = ClassificationService()
            self.classification_service.train(self.historical_data)
            
            # Initialize adaptive system (includes risk assessment and decision engine)
            self.adaptive_system = AdaptiveBotSystem()
            self.adaptive_system.train(self.historical_data)
            
            print("✅ All services initialized successfully")
            
        except Exception as e:
            print(f"❌ Error initializing services: {e}")
            raise
    
    def test_classification_performance(self, sample_size: int = 200) -> pd.DataFrame:
        """Test classification service performance"""
        
        print(f"\n🎯 Testing Classification Performance ({sample_size} samples)...")
        
        # Prepare test cases
        test_cases = self._prepare_classification_test_cases(sample_size)
        results = []
        
        for i, (text, true_type) in enumerate(test_cases):
            if i % 50 == 0:
                print(f"   Classification test {i+1}/{len(test_cases)}...")
            
            start_time = time.time()
            predicted_type, confidence = self.classification_service.classify_request(text)
            processing_time = (time.time() - start_time) * 1000
            
            result = {
                'text': text[:100],
                'true_type': true_type.value,
                'predicted_type': predicted_type.value,
                'confidence': confidence,
                'processing_time_ms': processing_time,
                'correct': predicted_type == true_type,
                'text_length': len(text),
                'word_count': len(text.split())
            }
            results.append(result)
            self.test_results['classification'].append(result)
        
        df = pd.DataFrame(results)
        accuracy = df['correct'].mean()
        avg_confidence = df['confidence'].mean()
        avg_time = df['processing_time_ms'].mean()
        
        print(f"   ✅ Classification Accuracy: {accuracy:.1%}")
        print(f"   📊 Average Confidence: {avg_confidence:.2f}")
        print(f"   ⏱️ Average Processing Time: {avg_time:.1f}ms")
        
        return df
    
    def test_risk_assessment_performance(self, sample_size: int = 200) -> pd.DataFrame:
        """Test risk assessment model performance"""
        
        print(f"\n🎲 Testing Risk Assessment Performance ({sample_size} samples)...")
        
        # Use historical data for testing
        test_data = self.historical_data.sample(min(sample_size, len(self.historical_data)))
        results = []
        
        # Get risk thresholds from the system
        risk_thresholds = self.adaptive_system.risk_assessment.learned_thresholds
        
        print("\n   📊 Risk Assessment Thresholds:")
        print(f"      Auto-Approval: Risk Score < {risk_thresholds['approval_threshold']}")
        print(f"      Manual Review: Risk Score >= {risk_thresholds['approval_threshold']} and < {risk_thresholds['rejection_threshold']}")
        print(f"      Auto-Reject: Risk Score >= {risk_thresholds['rejection_threshold']}")
        print()
        
        for i, row in test_data.iterrows():
            if len(results) % 50 == 0:
                print(f"   Risk assessment test {len(results)+1}/{len(test_data)}...")
            
            try:
                # Get true risk score from data
                true_risk = row['security_risk_score']
                request_text = row['request_summary']
                request_type = row['request_type']
                
                # Predict risk using the system
                start_time = time.time()
                predicted_risk, risk_info = self.adaptive_system.risk_assessment.assess_risk(
                    request_text, request_type
                )
                processing_time = (time.time() - start_time) * 1000
                
                # Calculate metrics
                risk_error = abs(predicted_risk - true_risk)
                risk_error_pct = risk_error / true_risk * 100
                
                # Determine accuracy (within 1.0 point is considered accurate)
                accurate = risk_error <= 1.0
                
                result = {
                    'text': request_text[:100],
                    'request_type': request_type,
                    'true_risk': true_risk,
                    'predicted_risk': predicted_risk,
                    'risk_error': risk_error,
                    'risk_error_pct': risk_error_pct,
                    'accurate': accurate,
                    'confidence': risk_info.get('confidence', 0.0),
                    'method': risk_info.get('method', 'unknown'),
                    'processing_time_ms': processing_time
                }
                results.append(result)
                self.test_results['risk_assessment'].append(result)
                
            except Exception as e:
                print(f"   ⚠️ Error in risk assessment test: {e}")
                continue
        
        df = pd.DataFrame(results)
        if not df.empty:
            accuracy = df['accurate'].mean()
            avg_error = df['risk_error'].mean()
            avg_time = df['processing_time_ms'].mean()
            
            print(f"   ✅ Risk Assessment Accuracy (±1.0): {accuracy:.1%}")
            print(f"   📊 Average Error: {avg_error:.2f} points")
            print(f"   ⏱️ Average Processing Time: {avg_time:.1f}ms")
            
            # Print sample of predictions
            print("\n   📈 Sample Predictions:")
            sample_size = min(5, len(df))
            sample_df = df.sample(sample_size)
            for _, row in sample_df.iterrows():
                print(f"      Request: {row['text']}")
                print(f"      True Risk: {row['true_risk']:.1f}, Predicted: {row['predicted_risk']:.1f}")
                print(f"      Error: {row['risk_error']:.1f} points ({row['risk_error_pct']:.1f}%)")
                print(f"      Confidence: {row['confidence']:.1%}")
                print()
        
        return df
    
    def test_decision_engine_performance(self, sample_size: int = 200) -> pd.DataFrame:
        """Test decision engine performance"""
        
        print(f"\n🤖 Testing Decision Engine Performance ({sample_size} samples)...")
        
        # Use historical data for testing
        test_data = self.historical_data.sample(min(sample_size, len(self.historical_data)))
        results = []
        
        # Get risk thresholds from the system
        risk_thresholds = self.adaptive_system.risk_assessment.learned_thresholds
        
        print("\n   📊 Decision Engine Thresholds:")
        print(f"      Auto-Approval: Risk Score < {risk_thresholds['approval_threshold']}")
        print(f"      Manual Review: Risk Score >= {risk_thresholds['approval_threshold']} and < {risk_thresholds['rejection_threshold']}")
        print(f"      Auto-Reject: Risk Score >= {risk_thresholds['rejection_threshold']}")
        print()
        
        for i, row in test_data.iterrows():
            if len(results) % 50 == 0:
                print(f"   Decision engine test {len(results)+1}/{len(test_data)}...")
            
            try:
                # Get true outcome from data
                true_outcome = row['outcome']
                request_text = row['request_summary']
                request_type = row['request_type']
                risk_score = row['security_risk_score']
                
                # Create request object
                request = SecurityRequest(
                    user_id="test_user",
                    channel_id="test_channel",
                    thread_ts="",
                    request_text=request_text,
                    request_type=RequestType(request_type),
                    risk_score=risk_score
                )
                
                # Get decision using the system
                start_time = time.time()
                predicted_outcome, decision_info = self.adaptive_system.decision_engine.make_decision(request)
                processing_time = (time.time() - start_time) * 1000
                
                # Calculate metrics
                correct = predicted_outcome.value == true_outcome
                
                # Ensure decision_info is a dictionary
                if isinstance(decision_info, str):
                    decision_info = {'rationale': decision_info}
                
                result = {
                    'text': request_text[:100],
                    'request_type': request_type,
                    'true_outcome': true_outcome,
                    'predicted_outcome': predicted_outcome.value,
                    'risk_score': risk_score,
                    'correct': correct,
                    'confidence': decision_info.get('confidence', 0.0),
                    'rationale': decision_info.get('rationale', ''),
                    'processing_time_ms': processing_time
                }
                results.append(result)
                self.test_results['decision_engine'].append(result)
                
            except Exception as e:
                print(f"   ⚠️ Error in decision engine test: {e}")
                continue
        
        df = pd.DataFrame(results)
        if not df.empty:
            accuracy = df['correct'].mean()
            avg_time = df['processing_time_ms'].mean()
            
            print(f"   ✅ Decision Engine Accuracy: {accuracy:.1%}")
            print(f"   ⏱️ Average Processing Time: {avg_time:.1f}ms")
            
            # Print sample of predictions
            print("\n   📈 Sample Predictions:")
            sample_size = min(5, len(df))
            sample_df = df.sample(sample_size)
            for _, row in sample_df.iterrows():
                print(f"      Request: {row['text']}")
                print(f"      True Outcome: {row['true_outcome']}, Predicted: {row['predicted_outcome']}")
                print(f"      Risk Score: {row['risk_score']:.1f}")
                print(f"      Confidence: {row['confidence']:.1%}")
                print(f"      Rationale: {row['rationale']}")
                print()
        
        return df
    
    def test_end_to_end_performance(self, sample_size: int = 100) -> pd.DataFrame:
        """Test complete end-to-end system performance"""
        
        print(f"\n🎪 Testing End-to-End System Performance ({sample_size} samples)...")
        
        test_data = self.historical_data.sample(min(sample_size, len(self.historical_data)))
        results = []
        
        for i, row in test_data.iterrows():
            if len(results) % 25 == 0:
                print(f"   End-to-end test {len(results)+1}/{len(test_data)}...")
            
            try:
                # True values
                true_type = RequestType(row['request_type'])
                true_risk = row['security_risk_score']
                true_outcome = Outcome(row['outcome'])
                
                # Test complete pipeline
                start_time = time.time()
                
                # Step 1: Classification
                predicted_type, class_confidence = self.classification_service.classify_request(
                    row['request_summary']
                )
                
                # Step 2: Create request object
                request = SecurityRequest(
                    user_id="test_user",
                    channel_id="test_channel", 
                    thread_ts="",
                    request_text=row['request_summary'],
                    request_type=predicted_type  # Use predicted type
                )
                
                # Step 3: Process with adaptive system
                processed_request, processing_info = self.adaptive_system.process_request(request)
                
                total_time = (time.time() - start_time) * 1000
                
                # Calculate metrics
                class_correct = predicted_type == true_type
                risk_error = abs(processed_request.risk_score - true_risk)
                risk_accurate = risk_error <= 1.0
                decision_correct = processed_request.outcome == true_outcome
                
                # Overall success (all components correct)
                overall_success = class_correct and risk_accurate and decision_correct
                
                result = {
                    'text': row['request_summary'][:100],
                    'true_type': true_type.value,
                    'predicted_type': predicted_type.value,
                    'class_correct': class_correct,
                    'class_confidence': class_confidence,
                    'true_risk': true_risk,
                    'predicted_risk': processed_request.risk_score,
                    'risk_error': risk_error,
                    'risk_accurate': risk_accurate,
                    'true_outcome': true_outcome.value,
                    'predicted_outcome': processed_request.outcome.value,
                    'decision_correct': decision_correct,
                    'overall_success': overall_success,
                    'total_time_ms': total_time,
                    'rationale': processed_request.rationale[:200]
                }
                results.append(result)
                self.test_results['end_to_end'].append(result)
                
            except Exception as e:
                print(f"   ⚠️ Error in end-to-end test: {e}")
                continue
        
        df = pd.DataFrame(results)
        if not df.empty:
            class_acc = df['class_correct'].mean()
            risk_acc = df['risk_accurate'].mean() 
            decision_acc = df['decision_correct'].mean()
            overall_acc = df['overall_success'].mean()
            avg_time = df['total_time_ms'].mean()
            
            print(f"   ✅ Classification Accuracy: {class_acc:.1%}")
            print(f"   ✅ Risk Assessment Accuracy: {risk_acc:.1%}")
            print(f"   ✅ Decision Engine Accuracy: {decision_acc:.1%}")
            print(f"   🎯 Overall Success Rate: {overall_acc:.1%}")
            print(f"   ⏱️ Average Total Time: {avg_time:.1f}ms")
        
        return df
    
    def _prepare_classification_test_cases(self, sample_size: int):
        """Prepare test cases for classification"""
        
        test_cases = []
        
        # High-quality custom test cases
        custom_cases = [
            ("Install Docker Desktop for containerized development", RequestType.DEVTOOL_INSTALL),
            ("Setup Visual Studio Code with Python extensions", RequestType.DEVTOOL_INSTALL),
            ("Download Java JDK 11 for enterprise applications", RequestType.DEVTOOL_INSTALL),
            
            ("Need admin access to production database server", RequestType.PERMISSION_CHANGE),
            ("Request elevated permissions for system maintenance", RequestType.PERMISSION_CHANGE),
            ("Temporary sudo access for critical security patching", RequestType.PERMISSION_CHANGE),
            
            ("Open port 443 for external API access", RequestType.NETWORK_ACCESS),
            ("Configure firewall rule for SSH access to development servers", RequestType.NETWORK_ACCESS),
            ("Allow VPN connection from remote office", RequestType.NETWORK_ACCESS),
            
            ("SSH access to AWS EC2 production instance", RequestType.CLOUD_RESOURCE_ACCESS),
            ("Access to Azure virtual machine for deployment", RequestType.CLOUD_RESOURCE_ACCESS),
            ("GCP compute engine access for monitoring", RequestType.CLOUD_RESOURCE_ACCESS),
            
            ("Export customer data for compliance audit", RequestType.DATA_EXPORT),
            ("Access to sales database for quarterly analysis", RequestType.DATA_EXPORT),
            ("Query user logs for security investigation", RequestType.DATA_EXPORT),
            
            ("Onboard vendor Salesforce for CRM integration", RequestType.VENDOR_APPROVAL),
            ("Approve contractor for penetration testing", RequestType.VENDOR_APPROVAL),
            ("Third-party analytics platform approval", RequestType.VENDOR_APPROVAL),
            
            ("Update firewall rules for new microservice", RequestType.FIREWALL_CHANGE),
            ("Modify AWS security group for load balancer", RequestType.FIREWALL_CHANGE),
            ("Configure iptables for container networking", RequestType.FIREWALL_CHANGE),
        ]
        
        test_cases.extend(custom_cases)
        
        # Add samples from historical data
        try:
            remaining_samples = sample_size - len(custom_cases)
            if remaining_samples > 0:
                historical_sample = self.historical_data.sample(min(remaining_samples, len(self.historical_data)))
                for _, row in historical_sample.iterrows():
                    try:
                        request_type = RequestType(row['request_type'])
                        test_cases.append((row['request_summary'], request_type))
                    except ValueError:
                        continue
        except Exception as e:
            print(f"   ⚠️ Error loading historical data: {e}")
        
        return test_cases[:sample_size]
    
    def generate_comprehensive_report(self):
        """Generate comprehensive performance report"""
        
        print("\n📊 GENERATING COMPREHENSIVE PERFORMANCE REPORT...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {},
            'detailed_results': self.test_results
        }
        
        # Calculate summary statistics
        for component, results in self.test_results.items():
            if not results:
                continue
                
            df = pd.DataFrame(results)
            
            if component == 'classification':
                accuracy = df['correct'].mean() if 'correct' in df.columns else 0
                avg_confidence = df['confidence'].mean() if 'confidence' in df.columns else 0
                avg_time = df['processing_time_ms'].mean() if 'processing_time_ms' in df.columns else 0
                
                report['summary'][component] = {
                    'accuracy': accuracy,
                    'average_confidence': avg_confidence,
                    'average_time_ms': avg_time,
                    'total_tests': len(results)
                }
                
            elif component == 'risk_assessment':
                accuracy = df['accurate'].mean() if 'accurate' in df.columns else 0
                avg_error = df['risk_error'].mean() if 'risk_error' in df.columns else 0
                avg_time = df['processing_time_ms'].mean() if 'processing_time_ms' in df.columns else 0
                
                report['summary'][component] = {
                    'accuracy': accuracy,
                    'average_error': avg_error,
                    'average_time_ms': avg_time,
                    'total_tests': len(results)
                }
                
            elif component == 'decision_engine':
                accuracy = df['correct'].mean() if 'correct' in df.columns else 0
                avg_time = df['processing_time_ms'].mean() if 'processing_time_ms' in df.columns else 0
                
                report['summary'][component] = {
                    'accuracy': accuracy,
                    'average_time_ms': avg_time,
                    'total_tests': len(results)
                }
                
            elif component == 'end_to_end':
                class_acc = df['class_correct'].mean() if 'class_correct' in df.columns else 0
                risk_acc = df['risk_accurate'].mean() if 'risk_accurate' in df.columns else 0
                decision_acc = df['decision_correct'].mean() if 'decision_correct' in df.columns else 0
                overall_acc = df['overall_success'].mean() if 'overall_success' in df.columns else 0
                avg_time = df['total_time_ms'].mean() if 'total_time_ms' in df.columns else 0
                
                report['summary'][component] = {
                    'classification_accuracy': class_acc,
                    'risk_accuracy': risk_acc,
                    'decision_accuracy': decision_acc,
                    'overall_accuracy': overall_acc,
                    'average_time_ms': avg_time,
                    'total_tests': len(results)
                }
        
        # Save report
        with open('model_performance_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Generate summary table
        summary_data = []
        for component, stats in report['summary'].items():
            if component == 'end_to_end':
                summary_data.append([
                    component.replace('_', ' ').title(),
                    stats['total_tests'],
                    f"{stats['overall_accuracy']:.1%}",
                    f"{stats['average_time_ms']:.1f}ms"
                ])
            else:
                accuracy_key = 'accuracy' if 'accuracy' in stats else 'overall_accuracy'
                summary_data.append([
                    component.replace('_', ' ').title(),
                    stats['total_tests'],
                    f"{stats[accuracy_key]:.1%}",
                    f"{stats['average_time_ms']:.1f}ms"
                ])
        
        table = tabulate(
            summary_data,
            headers=['Component', 'Tests', 'Accuracy', 'Avg Time'],
            tablefmt='grid'
        )
        
        print(f"\n🎯 PERFORMANCE SUMMARY:")
        print(table)
        
        return report
    
    def create_performance_visualizations(self):
        """Create comprehensive performance visualizations"""
        
        print("\n📈 GENERATING PERFORMANCE VISUALIZATIONS...")
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Model Performance Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Component Accuracy Comparison
        components = []
        accuracies = []
        
        for component, results in self.test_results.items():
            if not results:
                continue
                
            df = pd.DataFrame(results)
            
            if component == 'classification' and 'correct' in df.columns:
                components.append('Classification')
                accuracies.append(df['correct'].mean())
            elif component == 'risk_assessment' and 'accurate' in df.columns:
                components.append('Risk Assessment')
                accuracies.append(df['accurate'].mean())
            elif component == 'decision_engine' and 'correct' in df.columns:
                components.append('Decision Engine')
                accuracies.append(df['correct'].mean())
            elif component == 'end_to_end' and 'overall_success' in df.columns:
                components.append('End-to-End')
                accuracies.append(df['overall_success'].mean())
        
        if components and accuracies:
            bars = axes[0,0].bar(components, accuracies, color=['#2E8B57', '#4169E1', '#FF6347', '#FFD700'])
            axes[0,0].set_title('Model Accuracy Comparison', fontweight='bold')
            axes[0,0].set_ylabel('Accuracy')
            axes[0,0].set_ylim(0, 1)
            
            # Add value labels
            for i, v in enumerate(accuracies):
                axes[0,0].text(i, v + 0.01, f'{v:.1%}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Risk Assessment Error Distribution
        if self.test_results['risk_assessment']:
            risk_df = pd.DataFrame(self.test_results['risk_assessment'])
            if 'risk_error' in risk_df.columns:
                axes[0,1].hist(risk_df['risk_error'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                axes[0,1].set_title('Risk Assessment Error Distribution', fontweight='bold')
                axes[0,1].set_xlabel('Absolute Error')
                axes[0,1].set_ylabel('Frequency')
                axes[0,1].axvline(risk_df['risk_error'].mean(), color='red', linestyle='--', 
                                 label=f'Mean: {risk_df["risk_error"].mean():.2f}')
                axes[0,1].legend()
        
        # 3. Processing Time Comparison
        times = []
        time_labels = []
        
        for component, results in self.test_results.items():
            if not results:
                continue
                
            df = pd.DataFrame(results)
            time_col = 'total_time_ms' if component == 'end_to_end' else 'processing_time_ms'
            
            if time_col in df.columns:
                times.append(df[time_col].mean())
                time_labels.append(component.replace('_', ' ').title())
        
        if times and time_labels:
            axes[0,2].barh(time_labels, times, color=['#FFA500', '#20B2AA', '#9370DB', '#DC143C'])
            axes[0,2].set_title('Average Processing Time by Component', fontweight='bold')
            axes[0,2].set_xlabel('Time (ms)')
            
            # Add value labels
            for i, v in enumerate(times):
                axes[0,2].text(v + max(times) * 0.01, i, f'{v:.1f}ms', va='center', fontweight='bold')
        
        # 4. Decision Engine Confusion Matrix
        if self.test_results['decision_engine']:
            decision_df = pd.DataFrame(self.test_results['decision_engine'])
            if 'true_outcome' in decision_df.columns and 'predicted_outcome' in decision_df.columns:
                confusion_matrix = pd.crosstab(decision_df['true_outcome'], decision_df['predicted_outcome'])
                confusion_pct = confusion_matrix.div(confusion_matrix.sum(axis=1), axis=0) * 100
                
                sns.heatmap(confusion_pct, annot=True, fmt='.1f', cmap='Blues', 
                           ax=axes[1,0], cbar_kws={'label': 'Percentage (%)'})
                axes[1,0].set_title('Decision Engine Confusion Matrix', fontweight='bold')
                axes[1,0].set_xlabel('Predicted Outcome')
                axes[1,0].set_ylabel('True Outcome')
        
        # 5. End-to-End Success Rate by Request Type
        if self.test_results['end_to_end']:
            e2e_df = pd.DataFrame(self.test_results['end_to_end'])
            if 'predicted_type' in e2e_df.columns and 'overall_success' in e2e_df.columns:
                type_success = e2e_df.groupby('predicted_type')['overall_success'].mean().sort_values(ascending=False)
                
                axes[1,1].bar(range(len(type_success)), type_success.values, 
                             color=plt.cm.Set3(np.linspace(0, 1, len(type_success))))
                axes[1,1].set_title('Success Rate by Request Type', fontweight='bold')
                axes[1,1].set_ylabel('Success Rate')
                axes[1,1].set_xticks(range(len(type_success)))
                axes[1,1].set_xticklabels([t.replace('_', '\n') for t in type_success.index], 
                                         rotation=0, ha='center', fontsize=9)
                
                # Add value labels
                for i, v in enumerate(type_success.values):
                    axes[1,1].text(i, v + 0.01, f'{v:.1%}', ha='center', va='bottom', fontweight='bold')
        
        # 6. Confidence vs Accuracy (Classification)
        if self.test_results['classification']:
            class_df = pd.DataFrame(self.test_results['classification'])
            if 'confidence' in class_df.columns and 'correct' in class_df.columns:
                axes[1,2].scatter(class_df['confidence'], class_df['correct'].astype(int), 
                                 alpha=0.6, c=class_df['correct'].astype(int), cmap='RdYlGn')
                axes[1,2].set_title('Classification: Confidence vs Accuracy', fontweight='bold')
                axes[1,2].set_xlabel('Confidence Score')
                axes[1,2].set_ylabel('Correct (1) / Incorrect (0)')
                axes[1,2].set_ylim(-0.1, 1.1)
                
                # Add trend line
                z = np.polyfit(class_df['confidence'], class_df['correct'].astype(int), 1)
                p = np.poly1d(z)
                axes[1,2].plot(class_df['confidence'].sort_values(), 
                              p(class_df['confidence'].sort_values()), "r--", alpha=0.8, linewidth=2)
        
        plt.tight_layout()
        plt.savefig('model_performance_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Performance dashboard saved as 'model_performance_dashboard.png'")
    
    def run_complete_analysis(self, sample_size: int = 150):
        """Run complete model performance analysis"""
        
        print("🚀 STARTING COMPREHENSIVE MODEL PERFORMANCE ANALYSIS")
        print("=" * 70)
        
        try:
            # Test each component
            print("Testing individual components...")
            # self.test_classification_performance(sample_size)
            self.test_risk_assessment_performance(sample_size)
            self.test_decision_engine_performance(sample_size)
            
            print("\nTesting integrated system...")
            # self.test_end_to_end_performance(sample_size // 2)  # Smaller for end-to-end
            
            # Generate report and visualizations
            print("\nGenerating analysis...")
            report = self.generate_comprehensive_report()
            self.create_performance_visualizations()
            
            # Save detailed results
            for component, results in self.test_results.items():
                if results:
                    df = pd.DataFrame(results)
                    df.to_csv(f'{component}_test_results.csv', index=False)
            
            print("\n✅ ANALYSIS COMPLETE!")
            print("\nGenerated files:")
            print("   - model_performance_report.json")
            print("   - model_performance_dashboard.png")
            for component in self.test_results.keys():
                if self.test_results[component]:
                    print(f"   - {component}_test_results.csv")
            
            return report
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            raise

def main():
    """Main function"""
    
    print("🧪 MODEL PERFORMANCE TESTING SUITE")
    print("=" * 50)
    
    try:
        tester = ModelPerformanceTester()
        
        sample_size = input("Enter sample size for testing (default 150): ").strip()
        sample_size = int(sample_size) if sample_size.isdigit() else 150
        
        report = tester.run_complete_analysis(sample_size)
        
        print(f"\n🎯 FINAL SUMMARY:")
        print(f"   Overall system performance successfully analyzed")
        print(f"   Check generated files for detailed results")
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()