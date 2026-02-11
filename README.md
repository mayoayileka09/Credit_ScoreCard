# STREAM: Systematic Template for Reliable and Efficient Artificial intelligence and Machine learning

STREAM is a powerful, modular MLOps template designed to facilitate collaborative code development, automate processes, and ensure data traceability in machine learning workflows. Each workflow step is handled by a dedicated module, making the codebase easy to extend and maintain for future projects.

## Quickstart Demo: Using MLflow with STREAM

### 1. Prerequisites

- Python 3.11+
- MLflow server running (default: http://localhost:5000)
- All dependencies from `requirements.txt` installed

### 2. Project Setup

```sh
pip install -r requirements.txt
```

### 3. Configuration

Edit `ML_Project Credit_ScoreCard/config/config.py.jinja` to set your MLflow tracking URI and experiment name:

```python
MLFLOW_TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME = "demo_experiment"  # Use your existing experiment name
```

### 4. Run the Demo

Each CLI command in `main.py` delegates to a dedicated module for modularity and extensibility.

#### Preprocess

```sh
python ML_Project Credit_ScoreCard/main.py preprocess --test-run False
```

#### Train

```sh
python ML_Project Credit_ScoreCard/main.py train-cmd --test-run False
```

#### Process

```sh
python ML_Project Credit_ScoreCard/main.py process --test-run False
```

#### Evaluate (loads last run metric and plots)

```sh
python ML_Project Credit_ScoreCard/main.py evaluate --test-run False
```

### 5. Visualize Results

- The `evaluate` command uses `plotter.py` to plot regression results and save them as `eval_plot.png`.

### 6. MLflow UI

- Open MLflow UI to view experiments and runs:
```sh
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
- Visit [http://localhost:5000](http://localhost:5000) in your browser.

## Repository Structure

- `ML_Project Credit_ScoreCard/main.py`: CLI entry point for all workflow steps.
- `ML_Project Credit_ScoreCard/data.py`: Data preprocessing functions.
- `ML_Project Credit_ScoreCard/train.py`: Model training functions.
- `ML_Project Credit_ScoreCard/plotter.py`: Plotting functions for regression and metrics.
- `ML_Project Credit_ScoreCard/predict.py`: Prediction utilities (for extensibility).
- `ML_Project Credit_ScoreCard/config/`: Configuration files and MLflow settings.
- `ML_Project Credit_ScoreCard/tests/`: Unit tests for workflow steps.

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for more details.

## License

This project is licensed under the terms of the GPL license. See the [LICENSE](../LICENSE) file for the full license text.

## Contact

If you have any questions or feedback, please feel free to [contact us](mailto:ryangodwin@uab.edu).
