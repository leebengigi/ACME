#!/usr/bin/env python3
"""
Test script for the Ensemble Decision Engine
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# Add your project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import your ensemble engine
from src.services.adaptive_decision_engine import AdaptiveDecisionEngine
from src.models.enums import RequestType, Outcome
from src.models.data_models import SecurityRequest

def create_model_comparison_plots(model_scores, engine):
    """Create comprehensive model comparison visualizations"""
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create a large figure with subplots
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Model Performance Comparison Bar Chart
    plt.subplot(3, 3, 1)
    models = list(model_scores.keys())
    val_accuracies = [model_scores[m]['validation_accuracy'] for m in models]
    cv_accuracies = [model_scores[m]['cv_mean'] for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = plt.bar(x - width/2, val_accuracies, width, label='Validation Accuracy', alpha=0.8)
    bars2 = plt.bar(x + width/2, cv_accuracies, width, label='Cross-Validation Accuracy', alpha=0.8)
    
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.title('Model Performance Comparison', fontweight='bold')
    plt.xticks(x, [m.replace('_', ' ').title() for m in models], rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 2. Model Stability (CV Standard Deviation)
    plt.subplot(3, 3, 2)
    cv_stds = [model_scores[m]['cv_std'] for m in models]
    colors = plt.cm.Reds(np.linspace(0.4, 0.8, len(models)))
    
    bars = plt.bar(models, cv_stds, color=colors, alpha=0.8)
    plt.xlabel('Models')
    plt.ylabel('CV Standard Deviation')
    plt.title('Model Stability (Lower = More Stable)', fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 3. Combined Score Ranking
    plt.subplot(3, 3, 3)
    combined_scores = [model_scores[m]['combined_score'] for m in models]
    sorted_indices = np.argsort(combined_scores)[::-1]  # Descending order
    
    sorted_models = [models[i] for i in sorted_indices]
    sorted_scores = [combined_scores[i] for i in sorted_indices]
    
    colors = plt.cm.Greens(np.linspace(0.4, 0.9, len(models)))
    bars = plt.bar(range(len(sorted_models)), sorted_scores, color=colors, alpha=0.8)
    
    plt.xlabel('Model Rank')
    plt.ylabel('Combined Score')
    plt.title('Overall Model Ranking', fontweight='bold')
    plt.xticks(range(len(sorted_models)), [m.replace('_', ' ').title() for m in sorted_models], rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    # Add value labels and rank
    for i, bar in enumerate(bars):
        height = bar.get_height()
        rank = i + 1
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'#{rank}\n{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 4. Performance vs Stability Scatter Plot
    plt.subplot(3, 3, 4)
    val_accs = [model_scores[m]['validation_accuracy'] for m in models]
    cv_stds = [model_scores[m]['cv_std'] for m in models]
    
    plt.scatter(cv_stds, val_accs, s=100, alpha=0.7, c=range(len(models)), cmap='viridis')
    
    for i, model in enumerate(models):
        plt.annotate(model.replace('_', ' ').title(), 
                    (cv_stds[i], val_accs[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    plt.xlabel('CV Standard Deviation (Lower = More Stable)')
    plt.ylabel('Validation Accuracy (Higher = Better)')
    plt.title('Performance vs Stability Trade-off', fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add quadrant labels
    plt.axhline(np.mean(val_accs), color='red', linestyle='--', alpha=0.5)
    plt.axvline(np.mean(cv_stds), color='red', linestyle='--', alpha=0.5)
    
    # 5. Model Confidence Distribution (if available)
    plt.subplot(3, 3, 5)
    if hasattr(engine, 'models') and engine.models:
        # Create sample data to test model confidence
        sample_texts = [
            "Install Docker Desktop for development",
            "Need admin access to production database", 
            "Export customer data for compliance",
            "VPN access for remote work",
            "Critical security bypass needed"
        ]
        
        confidence_data = {}
        for model_name, model in engine.models.items():
            confidences = []
            for text in sample_texts:
                try:
                    # Create dummy request
                    from src.models.data_models import SecurityRequest
                    from src.models.enums import RequestType
                    
                    request = SecurityRequest(
                        user_id="test", channel_id="test", thread_ts="",
                        request_text=text, request_type=RequestType.OTHER, risk_score=50.0
                    )
                    
                    features = engine._extract_single_request_features(request)
                    if model_name == 'naive_bayes':
                        proba = model.predict_proba(features.toarray())[0]
                    else:
                        proba = model.predict_proba(features)[0]
                    
                    max_confidence = np.max(proba)
                    confidences.append(max_confidence)
                except:
                    confidences.append(0.5)  # Default confidence
            
            confidence_data[model_name] = confidences
        
        # Box plot of confidence distributions
        conf_values = list(confidence_data.values())
        conf_labels = [m.replace('_', ' ').title() for m in confidence_data.keys()]
        
        plt.boxplot(conf_values, labels=conf_labels)
        plt.xlabel('Models')
        plt.ylabel('Prediction Confidence')
        plt.title('Model Confidence Distribution', fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'Model confidence data\nnot available', 
                ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
        plt.title('Model Confidence Distribution', fontweight='bold')
    
    # 6. Feature Importance Comparison (for tree-based models)
    plt.subplot(3, 3, 6)
    if hasattr(engine, 'models') and 'random_forest' in engine.models:
        try:
            rf_model = engine.models['random_forest']
            if hasattr(rf_model, 'feature_importances_'):
                # Get top 10 features
                importances = rf_model.feature_importances_
                top_indices = np.argsort(importances)[-10:]
                top_importances = importances[top_indices]
                
                # Create feature names (simplified)
                feature_names = [f'Feature_{i}' for i in top_indices]
                
                plt.barh(range(len(top_importances)), top_importances, alpha=0.8)
                plt.yticks(range(len(top_importances)), feature_names)
                plt.xlabel('Feature Importance')
                plt.title('Top 10 Feature Importance\n(Random Forest)', fontweight='bold')
                plt.grid(True, alpha=0.3)
            else:
                plt.text(0.5, 0.5, 'Feature importance\nnot available', 
                        ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
                plt.title('Feature Importance Analysis', fontweight='bold')
        except:
            plt.text(0.5, 0.5, 'Feature importance\nanalysis failed', 
                    ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
            plt.title('Feature Importance Analysis', fontweight='bold')
    else:
        plt.text(0.5, 0.5, 'Feature importance\nnot available', 
                ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
        plt.title('Feature Importance Analysis', fontweight='bold')
    
    # 7. Model Prediction Agreement
    plt.subplot(3, 3, 7)
    if hasattr(engine, 'models') and len(engine.models) > 1:
        # Test agreement on sample data
        sample_predictions = {}
        sample_texts = [
            "Install Docker Desktop", "Admin access needed", "Export customer data",
            "VPN access request", "Firewall rule change", "Database query access"
        ]
        
        for model_name, model in engine.models.items():
            predictions = []
            for text in sample_texts:
                try:
                    request = SecurityRequest(
                        user_id="test", channel_id="test", thread_ts="",
                        request_text=text, request_type=RequestType.OTHER, risk_score=50.0
                    )
                    
                    features = engine._extract_single_request_features(request)
                    if model_name == 'naive_bayes':
                        pred = model.predict(features.toarray())[0]
                    else:
                        pred = model.predict(features)[0]
                    
                    predictions.append(pred)
                except:
                    predictions.append(0)  # Default prediction
            
            sample_predictions[model_name] = predictions
        
        # Calculate pairwise agreement
        model_names = list(sample_predictions.keys())
        agreement_matrix = np.zeros((len(model_names), len(model_names)))
        
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names):
                if i == j:
                    agreement_matrix[i, j] = 1.0
                else:
                    predictions1 = sample_predictions[model1]
                    predictions2 = sample_predictions[model2]
                    agreement = np.mean([p1 == p2 for p1, p2 in zip(predictions1, predictions2)])
                    agreement_matrix[i, j] = agreement
        
        sns.heatmap(agreement_matrix, 
                   annot=True, fmt='.2f', cmap='RdYlBu_r',
                   xticklabels=[m.replace('_', ' ').title() for m in model_names],
                   yticklabels=[m.replace('_', ' ').title() for m in model_names])
        plt.title('Model Prediction Agreement', fontweight='bold')
        plt.xlabel('Model')
        plt.ylabel('Model')
    else:
        plt.text(0.5, 0.5, 'Model agreement\nanalysis not available', 
                ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
        plt.title('Model Prediction Agreement', fontweight='bold')
    
    # 8. Training Time Comparison (simulated)
    plt.subplot(3, 3, 8)
    # Simulate training times based on model complexity
    training_times = {
        'naive_bayes': 0.1,
        'logistic_regression': 0.3,
        'random_forest': 2.5,
        'gradient_boosting': 4.2,
        'svm': 1.8
    }
    
    available_models = [m for m in models if m in training_times]
    times = [training_times.get(m, 1.0) for m in available_models]
    
    colors = plt.cm.Oranges(np.linspace(0.4, 0.8, len(available_models)))
    bars = plt.bar(available_models, times, color=colors, alpha=0.8)
    
    plt.xlabel('Models')
    plt.ylabel('Training Time (seconds)')
    plt.title('Estimated Training Time Comparison', fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.1f}s', ha='center', va='bottom', fontsize=9)
    
    # 9. Model Selection Summary
    plt.subplot(3, 3, 9)
    plt.axis('off')
    
    # Create summary text
    best_model = max(model_scores.keys(), key=lambda x: model_scores[x]['combined_score'])
    best_score = model_scores[best_model]['combined_score']
    best_accuracy = model_scores[best_model]['validation_accuracy']
    
    summary_text = f"""
    🏆 MODEL SELECTION SUMMARY
    
    Best Model: {best_model.replace('_', ' ').title()}
    
    📊 Performance:
    • Accuracy: {best_accuracy:.1%}
    • Combined Score: {best_score:.3f}
    
    📈 Key Insights:
    • {len(models)} models trained
    • Best stability: {min(model_scores.keys(), key=lambda x: model_scores[x]['cv_std']).replace('_', ' ').title()}
    • Highest accuracy: {max(model_scores.keys(), key=lambda x: model_scores[x]['validation_accuracy']).replace('_', ' ').title()}
    
    ✅ Recommendation:
    Use {best_model.replace('_', ' ').title()} for 
    optimal balance of accuracy 
    and stability.
    """
    
    plt.text(0.1, 0.9, summary_text, transform=plt.gca().transAxes, 
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    plt.suptitle('🤖 Comprehensive Model Comparison Dashboard', fontsize=16, fontweight='bold', y=0.98)
    
    # Save the comprehensive comparison
    plt.savefig('model_comparison_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create additional detailed comparison plots
    create_detailed_model_analysis(model_scores, engine)


def create_detailed_model_analysis(model_scores, engine):
    """Create additional detailed analysis plots"""
    
    # Create a separate figure for detailed analysis
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Radar Chart for Model Comparison
    ax1 = plt.subplot(2, 2, 1, projection='polar')
    
    models = list(model_scores.keys())[:5]  # Limit to 5 models for clarity
    
    # Metrics for radar chart
    metrics = ['Validation Acc', 'CV Mean', 'Stability', 'Combined Score']
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
    
    for i, model in enumerate(models):
        scores = model_scores[model]
        values = [
            scores['validation_accuracy'],
            scores['cv_mean'], 
            1 - scores['cv_std'],  # Invert std (higher is better)
            scores['combined_score']
        ]
        
        # Normalize values to 0-1 scale
        max_vals = [1.0, 1.0, 1.0, max(s['combined_score'] for s in model_scores.values())]
        normalized_values = [v / max_v for v, max_v in zip(values, max_vals)]
        normalized_values += normalized_values[:1]  # Complete the circle
        
        ax1.plot(angles, normalized_values, 'o-', linewidth=2, label=model.replace('_', ' ').title(), color=colors[i])
        ax1.fill(angles, normalized_values, alpha=0.25, color=colors[i])
    
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(metrics)
    ax1.set_ylim(0, 1)
    ax1.set_title('Model Performance Radar Chart', fontweight='bold', pad=20)
    ax1.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    # 2. Learning Curve Simulation
    plt.subplot(2, 2, 2)
    
    # Simulate learning curves based on model characteristics
    training_sizes = np.linspace(0.1, 1.0, 10)
    
    for model in models:
        if model == 'naive_bayes':
            # Fast learner, plateaus early
            curve = 0.7 + 0.15 * (1 - np.exp(-training_sizes * 8))
        elif model == 'random_forest':
            # Steady improvement
            curve = 0.5 + 0.35 * training_sizes**0.7
        elif model == 'gradient_boosting':
            # Best performance with enough data
            curve = 0.4 + 0.45 * training_sizes**0.5
        elif model == 'logistic_regression':
            # Linear improvement
            curve = 0.6 + 0.25 * training_sizes
        else:  # SVM
            # Slow start, good with more data
            curve = 0.45 + 0.35 * training_sizes**1.2
        
        # Add some noise
        curve += np.random.normal(0, 0.02, len(curve))
        
        plt.plot(training_sizes, curve, 'o-', label=model.replace('_', ' ').title(), linewidth=2, markersize=4)
    
    plt.xlabel('Training Set Size (proportion)')
    plt.ylabel('Validation Accuracy')
    plt.title('Simulated Learning Curves', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Model Complexity vs Performance
    plt.subplot(2, 2, 3)
    
    # Assign complexity scores to models
    complexity_scores = {
        'naive_bayes': 1,
        'logistic_regression': 2,
        'svm': 3,
        'random_forest': 4,
        'gradient_boosting': 5
    }
    
    complexities = [complexity_scores.get(m, 3) for m in models]
    accuracies = [model_scores[m]['validation_accuracy'] for m in models]
    stabilities = [1 - model_scores[m]['cv_std'] for m in models]  # Invert std
    
    # Create bubble chart
    plt.scatter(complexities, accuracies, s=[s*500 for s in stabilities], 
               alpha=0.6, c=range(len(models)), cmap='viridis')
    
    for i, model in enumerate(models):
        plt.annotate(model.replace('_', ' ').title(), 
                    (complexities[i], accuracies[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    plt.xlabel('Model Complexity (1=Simple, 5=Complex)')
    plt.ylabel('Validation Accuracy')
    plt.title('Complexity vs Performance\n(Bubble size = Stability)', fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 4. Model Strengths and Weaknesses
    plt.subplot(2, 2, 4)
    plt.axis('off')
    
    # Create strengths/weaknesses analysis
    model_analysis = {
        'random_forest': {'strengths': ['Feature interactions', 'Robust'], 'weaknesses': ['Can overfit', 'Black box']},
        'gradient_boosting': {'strengths': ['High performance', 'Feature importance'], 'weaknesses': ['Slow training', 'Complex tuning']},
        'logistic_regression': {'strengths': ['Interpretable', 'Fast'], 'weaknesses': ['Linear assumptions', 'Limited complexity']},
        'svm': {'strengths': ['Non-linear', 'Margin optimization'], 'weaknesses': ['Slow on large data', 'Parameter sensitive']},
        'naive_bayes': {'strengths': ['Fast', 'Simple'], 'weaknesses': ['Independence assumption', 'Limited performance']}
    }
    
    best_model = max(model_scores.keys(), key=lambda x: model_scores[x]['combined_score'])
    
    analysis_text = f"🎯 MODEL ANALYSIS\n\n"
    analysis_text += f"Best Model: {best_model.replace('_', ' ').title()}\n\n"
    
    if best_model in model_analysis:
        analysis_text += f"✅ Strengths:\n"
        for strength in model_analysis[best_model]['strengths']:
            analysis_text += f"  • {strength}\n"
        
        analysis_text += f"\n⚠️ Considerations:\n"
        for weakness in model_analysis[best_model]['weaknesses']:
            analysis_text += f"  • {weakness}\n"
    
    analysis_text += f"\n📊 Quick Comparison:\n"
    for model in models[:3]:  # Top 3 models
        score = model_scores[model]['validation_accuracy']
        analysis_text += f"  {model.replace('_', ' ').title()}: {score:.1%}\n"
    
    plt.text(0.05, 0.95, analysis_text, transform=plt.gca().transAxes, 
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
    
    plt.tight_layout()
    plt.suptitle('🔍 Detailed Model Analysis', fontsize=14, fontweight='bold', y=0.98)
    
    # Save detailed analysis
    plt.savefig('detailed_model_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


def create_confusion_matrix_analysis(y_true, y_pred, prediction_details):
    """Create comprehensive confusion matrix analysis"""
    
    # Get unique labels
    labels = sorted(list(set(y_true + y_pred)))
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    
    # Plot confusion matrix with percentages
    sns.heatmap(cm_percent, 
                annot=True, 
                fmt='.1f', 
                cmap='Blues',
                xticklabels=labels,
                yticklabels=labels,
                cbar_kws={'label': 'Percentage (%)'})
    
    plt.title('Decision Engine Confusion Matrix\n(Percentages)', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Outcome', fontsize=12)
    plt.ylabel('True Outcome', fontsize=12)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print detailed classification report
    print("\n📈 Detailed Classification Report:")
    print("=" * 60)
    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
    
    for label in labels:
        if label in report:
            precision = report[label]['precision']
            recall = report[label]['recall']
            f1_score = report[label]['f1-score']
            support = report[label]['support']
            
            print(f"{label:15}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1_score:.3f} (n={support})")
    
    # Overall metrics
    accuracy = report['accuracy']
    macro_avg = report['macro avg']
    weighted_avg = report['weighted avg']
    
    print(f"\n📊 Overall Metrics:")
    print(f"   Accuracy: {accuracy:.3f}")
    print(f"   Macro Avg:    P={macro_avg['precision']:.3f}, R={macro_avg['recall']:.3f}, F1={macro_avg['f1-score']:.3f}")
    print(f"   Weighted Avg: P={weighted_avg['precision']:.3f}, R={weighted_avg['recall']:.3f}, F1={weighted_avg['f1-score']:.3f}")
    
    # Raw confusion matrix
    print(f"\n🔢 Raw Confusion Matrix:")
    print("=" * 40)
    print(f"{'':15} {'Predicted →':>15}")
    print(f"{'True ↓':15} {' '.join(f'{label[:8]:>8}' for label in labels)}")
    print("-" * 40)
    
    for i, true_label in enumerate(labels):
        row_str = f"{true_label[:12]:12}   "
        for j, pred_label in enumerate(labels):
            row_str += f"{cm[i,j]:8d}"
        print(row_str)
    
    # Analysis of misclassifications
    print(f"\n🔍 Misclassification Analysis:")
    print("=" * 50)
    
    # Count misclassifications by type
    misclass_counts = {}
    for detail in prediction_details:
        if not detail['correct']:
            error_type = f"{detail['actual']} → {detail['predicted']}"
            if error_type not in misclass_counts:
                misclass_counts[error_type] = []
            misclass_counts[error_type].append(detail)
    
    if misclass_counts:
        for error_type, errors in sorted(misclass_counts.items(), key=lambda x: len(x[1]), reverse=True):
            print(f"\n❌ {error_type} ({len(errors)} cases):")
            
            # Show risk score distribution for this error type
            risk_scores = [e['risk_score'] for e in errors]
            if risk_scores:
                avg_risk = np.mean(risk_scores)
                min_risk = np.min(risk_scores)
                max_risk = np.max(risk_scores)
                print(f"   Risk scores: {min_risk:.1f}-{max_risk:.1f} (avg: {avg_risk:.1f})")
            
            # Show a few examples
            print("   Examples:")
            for error in errors[:3]:  # Show top 3
                print(f"   • Risk {error['risk_score']:.1f}: {error['text']}")
            
            if len(errors) > 3:
                print(f"   • ... and {len(errors)-3} more")
    else:
        print("🎉 No misclassifications found!")
    
    # Risk score analysis by outcome
    print(f"\n📈 Risk Score Analysis by Outcome:")
    print("=" * 45)
    
    risk_by_outcome = {}
    for detail in prediction_details:
        outcome = detail['actual']
        if outcome not in risk_by_outcome:
            risk_by_outcome[outcome] = []
        risk_by_outcome[outcome].append(detail['risk_score'])
    
    for outcome, risks in risk_by_outcome.items():
        if risks:
            avg_risk = np.mean(risks)
            std_risk = np.std(risks)
            min_risk = np.min(risks)
            max_risk = np.max(risks)
            count = len(risks)
            
            print(f"{outcome:15}: {count:3d} samples, risk {min_risk:5.1f}-{max_risk:5.1f} (μ={avg_risk:.1f}, σ={std_risk:.1f})")


def test_ensemble_engine():
    """Test the ensemble decision engine"""
    
    print("🚀 Testing Ensemble Decision Engine...")
    print("=" * 50)
    
    # Load your data
    try:
        df = pd.read_csv(r"C:\Users\leebg\Documents\Vyper\data\acme_security_tickets.csv")
        print(f"✅ Loaded {len(df)} tickets from CSV")
    except FileNotFoundError:
        print("❌ Could not find acme_security_tickets.csv")
        return
    
    # Create and train the ensemble
    engine = AdaptiveDecisionEngine()
    
    print("\n📚 Training Ensemble Models...")
    training_results = engine.train(df)
    
    if 'error' in training_results:
        print(f"❌ Training failed: {training_results['error']}")
        return
    
    # Display training results
    print(f"\n🎯 Training Results:")
    print(f"   Best Model: {training_results['best_model']}")
    print(f"   Best Accuracy: {training_results['best_accuracy']:.1%}")
    print(f"   Features Used: {training_results['feature_count']}")
    print(f"   Training Samples: {training_results['training_samples']}")
    
    print(f"\n🏆 All Model Scores:")
    for model_name, scores in training_results['all_model_scores'].items():
        val_acc = scores['validation_accuracy']
        cv_acc = scores['cv_mean']
        combined = scores['combined_score']
        print(f"   {model_name:20}: Val={val_acc:.3f}, CV={cv_acc:.3f}, Combined={combined:.3f}")
    
    # Create model comparison visualizations
    print(f"\n📊 Creating Model Comparison Visualizations...")
    create_model_comparison_plots(training_results['all_model_scores'], engine)
    
    # Test on some sample requests
    print(f"\n🧪 Testing Sample Predictions...")
    
    test_requests = [
        ("Install Docker Desktop for development", RequestType.DEVTOOL_INSTALL, 45.0),
        ("Need admin access to production database", RequestType.PERMISSION_CHANGE, 85.0),
        ("Export customer data for compliance report", RequestType.DATA_EXPORT, 75.0),
        ("VPN access for remote work", RequestType.NETWORK_ACCESS, 55.0),
        ("Onboard vendor Microsoft for analytics", RequestType.VENDOR_APPROVAL, 65.0),
    ]
    
    for i, (text, req_type, risk_score) in enumerate(test_requests, 1):
        request = SecurityRequest(
            user_id="test_user",
            channel_id="test_channel", 
            thread_ts="",
            request_text=text,
            request_type=req_type,
            risk_score=risk_score
        )
        
        outcome, rationale = engine.make_decision(request)
        print(f"   {i}. {text[:40]}...")
        print(f"      Risk: {risk_score:.1f} → {outcome.value}")
        print(f"      Rationale: {rationale}")
        print()
    
    # Validate on historical data
    print(f"🔍 Validating on Historical Data...")
    
    # Test on a subset of historical data
    test_sample = df.sample(min(200, len(df)), random_state=42)
    correct_predictions = 0
    total_predictions = 0
    
    # Store predictions for confusion matrix
    y_true = []
    y_pred = []
    prediction_details = []
    
    outcome_mapping = {
        'Approved': Outcome.APPROVED,
        'Rejected': Outcome.REJECTED, 
        'Info Requested': Outcome.NEEDS_MORE_INFO
    }
    
    for _, row in test_sample.iterrows():
        try:
            # Create request
            req_type = RequestType.OTHER  # Default
            try:
                req_type = RequestType(row['request_type'])
            except:
                pass
                
            request = SecurityRequest(
                user_id="validation",
                channel_id="validation",
                thread_ts="",
                request_text=row['request_summary'],
                request_type=req_type,
                risk_score=row['security_risk_score']
            )
            
            # Get prediction
            predicted_outcome, rationale = engine.make_decision(request)
            actual_outcome = outcome_mapping.get(row['outcome'])
            
            if actual_outcome:
                total_predictions += 1
                y_true.append(actual_outcome.value)
                y_pred.append(predicted_outcome.value)
                
                prediction_details.append({
                    'text': row['request_summary'][:50] + "...",
                    'risk_score': row['security_risk_score'],
                    'actual': actual_outcome.value,
                    'predicted': predicted_outcome.value,
                    'correct': predicted_outcome == actual_outcome
                })
                
                if predicted_outcome == actual_outcome:
                    correct_predictions += 1
            
        except Exception as e:
            print(f"   Warning: Validation error for row: {e}")
            continue
    
    if total_predictions > 0:
        validation_accuracy = correct_predictions / total_predictions
        print(f"   Validation Accuracy: {validation_accuracy:.1%} ({correct_predictions}/{total_predictions})")
        
        # Create and display confusion matrix
        print(f"\n📊 Confusion Matrix & Detailed Analysis:")
        create_confusion_matrix_analysis(y_true, y_pred, prediction_details)
        
    else:
        print(f"   Could not validate on historical data")
    
    print(f"\n✅ Ensemble Testing Complete!")
    return engine

if __name__ == "__main__":
    engine = test_ensemble_engine()