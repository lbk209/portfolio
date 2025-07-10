from ipywidgets import Checkbox, DatePicker, IntSlider, ToggleButton, Layout, VBox

def on_toggle(change):
    if change.new:
        w_save.description = 'Save'
    else:
        w_save.description = 'Don\'t Save'


def WidgetCheckbox(*labels, as_accessor=False):
    """
    Create a VBox of checkboxes labeled by *labels.

    Args:
        *labels (str): Checkbox labels.
        as_accessor (bool): If True, return a ValueAccessor instance wrapping the VBox.

    Returns:
        VBox or ValueAccessor: VBox of checkboxes, or ValueAccessor if as_accessor=True.
    """
    vbox = VBox([
        Checkbox(value=False, description=label, disabled=False, indent=False)
        for label in labels
    ])
    return ValueAccessor(vbox) if as_accessor else vbox
    

class ValueAccessor:
    """
    Helper class to access the .value property of widgets (like Checkbox or DatePicker)
    inside a container widget (e.g., VBox), using either index or label.

    Args:
        container (ipywidgets.Box): A container widget (like VBox or HBox) with children
                                    that have 'description' and 'value' attributes.

    Methods:
        __getitem__(key): Access widget value by index (int) or label (str).
        as_dict(): Return a dictionary of {description: value} for all child widgets.

    Raises:
        KeyError: If no widget matches the given label.
        TypeError: If key is not int or str.

    Example:
        va = ValueAccessor(vbox)
        va[0]               # Value of the first widget
        va['Option A']      # Value of the widget with description 'Option A'
        va.as_dict()        # All values in dict form
    """
    def __init__(self, container):
        self._container = container
        self._children = container.children

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._children[key].value
        elif isinstance(key, str):
            for widget in self._children:
                if hasattr(widget, 'description') and widget.description == key:
                    return widget.value
            raise KeyError(f"No widget with description '{key}'")
        else:
            raise TypeError("Key must be int or str")

    def as_dict(self):
        return {
            widget.description: widget.value
            for widget in self._children
            if hasattr(widget, 'description')
        }
    
    def _ipython_display_(self):
        from IPython.display import display
        display(self._container)

        
item_layout = Layout(width='200px')

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

WidgetUniverse = WidgetCheckbox('Download', 'Closed', 'Overwrite', 'Cleanup', as_accessor=True)

WidgetTransaction = VBox([w_date, w_cap, w_save])
WidgetTransaction = ValueAccessor(WidgetTransaction)
