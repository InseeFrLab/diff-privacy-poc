from app.poc import app_ui, server, www_dir
from shiny import App

app = App(app_ui, server, static_assets=www_dir)
