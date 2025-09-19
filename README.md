# Simple MLOps

This repository contains a simple example of implementation github action in MLOps principles. The project demonstrates how to set up a basic pipeline for training, validating, and push application into github package.

## Features

- Model training and validation
- Integration with FastAPI for serving the model
- Using MLflow for experiment tracking
  - Using Minio as an artifact store
  - Postgres as a backend store
- Pushing docker image into registry
- Scanning docker image for vulnerabilities
- Version control with DVC

## Note for self-hosted runner

[Issue Runner](https://github.com/actions/setup-python/issues/460)
