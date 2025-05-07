import json
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from plotly.offline import plot

# Function to load JSON data and extract metrics
def load_and_process_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    # Extract run data
    print(data['model'])
    if data['model'] == "ResGCN":
      run = data["record"][0]
    elif data['model'] == "ResGAT":
      run = data["record"][2]
    elif data['model'] == "ResGAT with BERT-Chinese":
      run = data["record"][4] 
      
      
    test_accs = run["test accs"]
    test_precs = run["test precs"]
    test_recs = run["test recs"]
    test_f1s = run["test f1s"]
    
    # Calculate average metrics
    avg_precs = [sum(p) / len(p) for p in test_precs]
    avg_recs = [sum(r) / len(r) for r in test_recs]
    avg_f1s = [sum(f) / len(f) for f in test_f1s]
    
    # Find best epoch
    combined_scores = [test_accs[i] + avg_precs[i] + avg_recs[i] + avg_f1s[i] for i in range(len(test_accs))]
    best_overall_epoch = np.argmax(combined_scores) + 1
    
    return {
        "model_info": data,
        "test_accs": test_accs,
        "test_precs": test_precs,
        "test_recs": test_recs,
        "test_f1s": test_f1s,
        "avg_precs": avg_precs,
        "avg_recs": avg_recs,
        "avg_f1s": avg_f1s,
        "best_epoch": best_overall_epoch,
        "epochs": list(range(1, len(test_accs) + 1))
    }

# Create performance visualization
def create_visualization(data, output_file="model_performance.html"):
    # Unpack data
    model_info = data["model_info"]
    model_name = model_info["model"]
    dataset_name = model_info["dataset"]
    test_accs = data["test_accs"]
    test_precs = data["test_precs"]
    test_recs = data["test_recs"]
    test_f1s = data["test_f1s"]
    avg_precs = data["avg_precs"]
    avg_recs = data["avg_recs"]
    avg_f1s = data["avg_f1s"]
    best_epoch = data["best_epoch"]
    epochs = data["epochs"]
    
    # Create a figure with all metrics
    fig1 = go.Figure()
    
    # Add traces for overall metrics
    fig1.add_trace(go.Scatter(x=epochs, y=test_accs, mode='lines', name='Accuracy', line=dict(color='blue')))
    fig1.add_trace(go.Scatter(x=epochs, y=avg_precs, mode='lines', name='Precision (Avg)', line=dict(color='green')))
    fig1.add_trace(go.Scatter(x=epochs, y=avg_recs, mode='lines', name='Recall (Avg)', line=dict(color='orange')))
    fig1.add_trace(go.Scatter(x=epochs, y=avg_f1s, mode='lines', name='F1 Score (Avg)', line=dict(color='purple')))
    
    # Add vertical line for best epoch
    fig1.add_vline(x=best_epoch, line_width=2, line_dash="dash", line_color="red",
                  annotation_text=f"Best Epoch: {best_epoch}", 
                  annotation_position="top right")
    
    # Update layout
    fig1.update_layout(
        title=f"{model_name} Overall Performance Metrics (Run 0)",
        xaxis_title="Epoch",
        yaxis_title="Score",
        yaxis=dict(range=[0.3, 1]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Create a figure with class-specific metrics
    fig2 = make_subplots(rows=2, cols=2, 
                       subplot_titles=("Class 0 Metrics", "Class 1 Metrics", 
                                       "Accuracy", "F1 Score Comparison"))
    
    # Add class 0 metrics
    fig2.add_trace(go.Scatter(x=epochs, y=[p[0] for p in test_precs], mode='lines', 
                            name='Precision (Class 0)', line=dict(color='green')), row=1, col=1)
    fig2.add_trace(go.Scatter(x=epochs, y=[r[0] for r in test_recs], mode='lines', 
                            name='Recall (Class 0)', line=dict(color='orange')), row=1, col=1)
    fig2.add_trace(go.Scatter(x=epochs, y=[f[0] for f in test_f1s], mode='lines', 
                            name='F1 (Class 0)', line=dict(color='purple')), row=1, col=1)
    
    # Add class 1 metrics
    fig2.add_trace(go.Scatter(x=epochs, y=[p[1] for p in test_precs], mode='lines', 
                            name='Precision (Class 1)', line=dict(color='green', dash='dash')), row=1, col=2)
    fig2.add_trace(go.Scatter(x=epochs, y=[r[1] for r in test_recs], mode='lines', 
                            name='Recall (Class 1)', line=dict(color='orange', dash='dash')), row=1, col=2)
    fig2.add_trace(go.Scatter(x=epochs, y=[f[1] for f in test_f1s], mode='lines', 
                            name='F1 (Class 1)', line=dict(color='purple', dash='dash')), row=1, col=2)
    
    # Add accuracy
    fig2.add_trace(go.Scatter(x=epochs, y=test_accs, mode='lines', 
                            name='Accuracy', line=dict(color='blue')), row=2, col=1)
    
    # Add F1 comparison
    fig2.add_trace(go.Scatter(x=epochs, y=[f[0] for f in test_f1s], mode='lines', 
                            name='F1 (Class 0)', line=dict(color='lightblue')), row=2, col=2)
    fig2.add_trace(go.Scatter(x=epochs, y=[f[1] for f in test_f1s], mode='lines', 
                            name='F1 (Class 1)', line=dict(color='pink')), row=2, col=2)
    fig2.add_trace(go.Scatter(x=epochs, y=avg_f1s, mode='lines', 
                            name='F1 (Avg)', line=dict(color='purple')), row=2, col=2)
    
    # Add vertical line for best epoch in all subplots
    for i in range(1, 3):
        for j in range(1, 3):
            fig2.add_vline(x=best_epoch, line_width=2, line_dash="dash", line_color="red", row=i, col=j)
    
    # Update layout
    fig2.update_layout(
        title=f"{model_name} Detailed Metrics by Class (Run 0)",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    
    # Create a table with best metrics
    fig3 = go.Figure()
    
    best_metrics_data = [
        ["Metric", "Value at Best Epoch"],
        ["Accuracy", f"{test_accs[best_epoch-1]:.4f}"],
        ["Precision (Avg)", f"{avg_precs[best_epoch-1]:.4f}"],
        ["Recall (Avg)", f"{avg_recs[best_epoch-1]:.4f}"],
        ["F1 Score (Avg)", f"{avg_f1s[best_epoch-1]:.4f}"],
        ["Precision (Class 0)", f"{test_precs[best_epoch-1][0]:.4f}"],
        ["Precision (Class 1)", f"{test_precs[best_epoch-1][1]:.4f}"],
        ["Recall (Class 0)", f"{test_recs[best_epoch-1][0]:.4f}"],
        ["Recall (Class 1)", f"{test_recs[best_epoch-1][1]:.4f}"],
        ["F1 (Class 0)", f"{test_f1s[best_epoch-1][0]:.4f}"],
        ["F1 (Class 1)", f"{test_f1s[best_epoch-1][1]:.4f}"]
    ]
    
    fig3.add_trace(go.Table(
        header=dict(
            values=[best_metrics_data[0][0], best_metrics_data[0][1]],
            fill_color='paleturquoise',
            align='center',
            font=dict(size=14)
        ),
        cells=dict(
            values=[
                [row[0] for row in best_metrics_data[1:]],
                [row[1] for row in best_metrics_data[1:]]
            ],
            fill_color='lavender',
            align='center',
            font=dict(size=12)
        )
    ))
    
    fig3.update_layout(
        title=f"Best Performance Metrics (Epoch {best_epoch})"
    )
    
    # Create a table with model configuration
    fig4 = go.Figure()
    
    model_config_data = [
        ["Parameter", "Value"],
        ["Model", model_info["model"]],
        ["Dataset", model_info["dataset"]],
        ["Train Size", model_info["unsup train size"]],
        ["Batch Size", model_info["batch size"]],
        ["Hidden", model_info["hidden"]],
        ["Vector Size", model_info["vector size"]],
        ["Layers", f"{model_info['n layers feat']} feat, {model_info['n layers conv']} conv, {model_info['n layers fc']} fc"],
        ["Dropout", model_info["dropout"]],
        ["Epochs", model_info["epochs"]],
        ["Best Epoch", best_epoch]
    ]
    
    fig4.add_trace(go.Table(
        header=dict(
            values=[model_config_data[0][0], model_config_data[0][1]],
            fill_color='paleturquoise',
            align='center',
            font=dict(size=14)
        ),
        cells=dict(
            values=[
                [row[0] for row in model_config_data[1:]],
                [row[1] for row in model_config_data[1:]]
            ],
            fill_color='lavender',
            align='center',
            font=dict(size=12)
        )
    ))
    
    fig4.update_layout(
        title="Model Configuration"
    )
    
    # Using fig.write_html() method to save each figure
    fig1_file = f"{output_file.replace('.html', '')}_overall_metrics.html"
    fig2_file = f"{output_file.replace('.html', '')}_detailed_metrics.html"
    fig3_file = f"{output_file.replace('.html', '')}_best_metrics.html"
    fig4_file = f"{output_file.replace('.html', '')}_config.html"
    
    fig1.write_html(fig1_file)
    fig2.write_html(fig2_file)
    fig3.write_html(fig3_file)
    fig4.write_html(fig4_file)
    
    # Create a single HTML file with all visualizations
    # This uses the iframes approach
    with open(output_file, 'w') as f:
        f.write(f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model Performance Visualization - {model_name}</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f8f9fa;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 20px;
                    box-shadow: 0 0 10px rgba(0,0,0,0.1);
                    border-radius: 8px;
                }}
                .title {{
                    text-align: center;
                    color: #333;
                    margin-bottom: 30px;
                }}
                .iframe-container {{
                    margin-bottom: 40px;
                }}
                .row {{
                    display: flex;
                    flex-wrap: wrap;
                    margin: 0 -15px;
                }}
                .col {{
                    flex: 1;
                    padding: 0 15px;
                    min-width: 300px;
                }}
                iframe {{
                    width: 100%;
                    border: none;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1 class="title">Model Performance Visualization - {model_name}</h1>
                
                <div class="row">
                    <div class="col">
                        <div class="iframe-container">
                            <h2>Model Configuration</h2>
                            <iframe src="{fig4_file}" height="500px"></iframe>
                        </div>
                    </div>
                    <div class="col">
                        <div class="iframe-container">
                            <h2>Best Performance Metrics</h2>
                            <iframe src="{fig3_file}" height="500px"></iframe>
                        </div>
                    </div>
                </div>
                
                <div class="iframe-container">
                    <h2>Overall Performance Metrics</h2>
                    <iframe src="{fig1_file}" height="600px"></iframe>
                </div>
                
                <div class="iframe-container">
                    <h2>Detailed Class Metrics</h2>
                    <iframe src="{fig2_file}" height="800px"></iframe>
                </div>
            </div>
        </body>
        </html>
        """)
    
    print(f"Visualization saved to {output_file}")
    print(f"Individual visualizations saved to:")
    print(f"  - {fig1_file}")
    print(f"  - {fig2_file}")
    print(f"  - {fig3_file}")
    print(f"  - {fig4_file}")
    
    return output_file

# Main script execution
if __name__ == "__main__":
    model_files = ["ResGCN.json", "ResGAT.json", "BERTGAT.json"]
    for i, file_path in enumerate(model_files):
        model_data = load_and_process_data(file_path)
        output_file = create_visualization(model_data, f"{model_data['model_info']['model'].lower()}_performance.html")
        print(f"Created visualization for {model_data['model_info']['model']}")