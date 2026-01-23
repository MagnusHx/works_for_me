# audio_emotion

We built an audio emotion recognition system that predicts emotions from short speech clips. Our pipeline starts with raw audio files, which we preprocess into consistent feature representations (stored as .npy files) so training and inference are fast and reproducible. We trained a neural network in PyTorch to classify emotions such as happy, sad, angry, etc., and then wrapped the trained model in a small API using BentoML. Finally, we containerized the service and deployed it to Google Cloud Run, so the model can be queried through HTTP requests. To make it practical to use with our dataset setup, the API can also run predictions directly on feature files stored in a Google Cloud Storage bucket

## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed.dvc
│
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
├── tasks.py                  # Project tasks
├── bentofile.yaml            # Build recipe for bento
├── profile.prof              # Profiling output
├── requirements.bento.txt    # Dependencies for bento
├── service.py                # Service APIs
├── docker-compose.yml.
└── docker_how_to.txt         # Explanation of how to run docker

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
