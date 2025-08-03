# Differentially private statistics

## Install project
```
uv venv statsdp
source statsdp/bin/activate
uv pip install -e .
```


## Run the POC application

Start the `app.poc` application using `shiny run`. Add appropriate options if needed.
For instance:
```bash
shiny run app.poc
shiny run app.poc --reload --launch-browser
shiny run app.poc --autoreload-port 8000
shiny run app.poc --port 5000 --host 0.0.0.0
```
