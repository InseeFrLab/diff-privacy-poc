# Differentially private statistics

## Install project
```
uv venv statsdp
source statsdp/bin/activate
uv pip install -e .
```


## Run the POC application

Start the `main_app` application using `shiny run`. Add appropriate options if needed.
For instance:
```bash
shiny run main_app
shiny run main_app --reload --launch-browser
shiny run main_app --autoreload-port 8000
shiny run main_app --port 5000 --host 0.0.0.0
```
