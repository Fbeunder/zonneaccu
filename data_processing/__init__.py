"""Package voor modules die verantwoordelijk zijn voor het laden en analyseren van energiedata.

Dit package bevat modules voor:
- Het inlezen van data uit verschillende bronnen (CSV, API, etc.)
- Het valideren en normaliseren van data
- Het uitvoeren van analyses op de data
- Het voorbereiden van data voor visualisatie en berekeningen
"""

from data_processing import data_loader, data_analysis

__all__ = ['data_loader', 'data_analysis']