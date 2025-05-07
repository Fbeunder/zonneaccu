"""Package voor verschillende energieopslagmodules.

Dit package bevat modules voor het berekenen van de effectiviteit en rendabiliteit
van verschillende energieopslagmethoden zoals:
- Warmwaterboilers
- Accu's/Batterijen
- Andere opslagmethoden die in de toekomst kunnen worden toegevoegd
"""

from storage_modules import boiler_module

__all__ = ['boiler_module']
