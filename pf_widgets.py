from ipywidgets import Checkbox, DatePicker, IntSlider, ToggleButton, Layout, VBox

def on_toggle(change):
    if change.new:
        w_save.description = 'Save'
    else:
        w_save.description = 'Don\'t Save'
        
item_layout = Layout(width='200px')

w_download = Checkbox(
    value=False,
    description='Download',
    disabled=False,
    indent=False
)

w_close = Checkbox(
    value=False,
    description='Closed',
    disabled=False,
    indent=False
)

w_overwrite = Checkbox(
    value=False,
    description='Overwrite',
    disabled=False,
    indent=False
)

w_date = DatePicker(
    #description='Date',
    value=None,
    disabled=False,
    layout=item_layout
)

w_cap = IntSlider(
    #description='Capital',
    value=0,
    min=0,
    max=30000000,
    step=500000,
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format=',d',
    layout=item_layout
)

w_save = ToggleButton(
    description='Don\'t Save',
    value=False,
    disabled=False,
    button_style='', # 'success', 'info', 'warning', 'danger' or ''
    tooltip='Description',
    icon='save', # (FontAwesome names without the `fa-` prefix)
    layout=item_layout,
)
w_save.observe(on_toggle, names='value')

WidgetUniverse = VBox([w_download, w_close, w_overwrite])
WidgetTransaction = VBox([w_date, w_cap, w_save])

WidgetUniverse.values = lambda i: WidgetUniverse.children[i].value
WidgetTransaction.values = lambda i: WidgetTransaction.children[i].value