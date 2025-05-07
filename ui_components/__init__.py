"""UI componenten voor de Zonneaccu applicatie."""

from ui_components.data_import_page import render_data_import_page
from ui_components.data_analysis_page import render_data_analysis_page
from ui_components.boiler_analysis_page import render_boiler_analysis_page
from ui_components.battery_analysis_page import render_battery_analysis_page

# Niet direct importeren om circulaire imports te voorkomen
# De modules worden direct geïmporteerd in app.py