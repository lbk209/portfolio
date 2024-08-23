import ipywidgets as widgets

w_download = widgets.Checkbox(
    value=False,
    description='Download',
    disabled=False,
    indent=False
)

w_close = widgets.Checkbox(
    value=False,
    description='Closed',
    disabled=False,
    indent=False
)