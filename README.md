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

Start the `main_app` application using `shiny run`. Add appropriate options if needed.
For instance:
```bash
shiny run main_app
shiny run main_app --reload --launch-browser
shiny run main_app --autoreload-port 8000
shiny run main_app --port 5000 --host 0.0.0.0
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

To (re)generate Sphinx configuration
```bash
sphinx-apidoc  -o docs/source/ src/stats_dp/ --separate
```

To (re)generate documentation
```bash
make -C docs clean html
```

To browse generated documentation:
```bash
python -m http.server -d docs/_build/html
```
then access (http://localhost:8000)[http://localhost:8000] from a web browser.
