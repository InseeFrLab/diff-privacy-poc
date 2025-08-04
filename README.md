# Differentially private statistics

## Install the project
For local installation in current environment
```bash
uv pip install .
```

For global installation
```bash
uv pip install . --system
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

## Install project for development

```bash
uv venv statsdp
source statsdp/bin/activate
uv pip install -e .
```


To use in notebooks, avoid `uv`:
```bash
python -m venv create statsdp
source statsdp/bin/activate
python -m ensurepip --upgrade
python -m pip install -e .
python -m pip install ipykernel
python -m ipykernel install --user --name=statsdp --display-name "Python (statsdp)"
```



## Generate documentation

First install requirements
```bash
uv pip install -r requirements-docs.txt
```

To (re)generate documentation, run `make html` from the `/docs` folder

To browse generated documentation:
```bash
uv python -m http.server -d docs/_build/html
```
then access (http://localhost:8000)[http://localhost:8000] from a web browser.
