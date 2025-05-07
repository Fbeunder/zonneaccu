"""Module voor het beheren van configuratie en gebruikersinstellingen.

Deze module biedt functionaliteiten voor:
- Het laden en opslaan van configuratie-instellingen
- Het beheren van gebruikersvoorkeuren
- Het valideren van configuratiewaarden
- Het bieden van standaardwaarden voor verschillende instellingen
"""

import json
import os
import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

# Standaard configuratiewaarden
DEFAULT_CONFIG = {
    "boiler": {
        "capacity_liters": 200,
        "temperature_min": 40,
        "temperature_max": 80,
        "efficiency": 0.95,
    },
    "battery": {
        "capacity_kwh": 5,
        "max_power_kw": 2.5,
        "efficiency_charge": 0.92,
        "efficiency_discharge": 0.92,
        "depth_of_discharge": 0.8,
    },
    "energy": {
        "electricity_price": 0.28,  # Euro per kWh
        "gas_price": 1.10,  # Euro per m3
        "feed_in_tariff": 0.08,  # Euro per kWh
    }
}

# Configuratie schema voor validatie
CONFIG_SCHEMA = {
    "boiler": {
        "capacity_liters": {"type": "float", "min": 50, "max": 1000},
        "temperature_min": {"type": "float", "min": 20, "max": 60},
        "temperature_max": {"type": "float", "min": 40, "max": 95},
        "efficiency": {"type": "float", "min": 0.5, "max": 1.0},
    },
    "battery": {
        "capacity_kwh": {"type": "float", "min": 0.5, "max": 100},
        "max_power_kw": {"type": "float", "min": 0.5, "max": 20},
        "efficiency_charge": {"type": "float", "min": 0.5, "max": 1.0},
        "efficiency_discharge": {"type": "float", "min": 0.5, "max": 1.0},
        "depth_of_discharge": {"type": "float", "min": 0.1, "max": 1.0},
    },
    "energy": {
        "electricity_price": {"type": "float", "min": 0.01, "max": 1.0},
        "gas_price": {"type": "float", "min": 0.1, "max": 5.0},
        "feed_in_tariff": {"type": "float", "min": 0.0, "max": 1.0},
    }
}

# Configuratiemap
CONFIG_DIR = Path("configs")


class ConfigManager:
    """Klasse voor het beheren van configuraties."""
    
    def __init__(self, config_dir: Optional[Path] = None):
        """Initialiseer de ConfigManager.
        
        Args:
            config_dir (Optional[Path]): Map waar configuraties worden opgeslagen.
                Als None, wordt de standaard map gebruikt.
        """
        self.config_dir = config_dir or CONFIG_DIR
        self.ensure_config_dir()
        
    def ensure_config_dir(self) -> None:
        """Zorg ervoor dat de configuratiemap bestaat."""
        if not self.config_dir.exists():
            self.config_dir.mkdir(parents=True, exist_ok=True)
    
    def get_config_list(self) -> List[str]:
        """Haal een lijst van beschikbare configuratieprofielen op.
        
        Returns:
            List[str]: Lijst met namen van beschikbare configuratieprofielen.
        """
        self.ensure_config_dir()
        config_files = list(self.config_dir.glob("*.json"))
        return [f.stem for f in config_files]
    
    def load_config(self, profile_name: Optional[str] = None) -> Dict[str, Any]:
        """Laad een configuratieprofiel.
        
        Args:
            profile_name (Optional[str]): Naam van het te laden profiel.
                Als None, worden de standaardwaarden gebruikt.
                
        Returns:
            Dict[str, Any]: De geladen configuratie.
        """
        if profile_name is None:
            return DEFAULT_CONFIG.copy()
        
        config_path = self.config_dir / f"{profile_name}.json"
        if not config_path.exists():
            return DEFAULT_CONFIG.copy()
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            return validate_config(config)
        except (json.JSONDecodeError, IOError):
            return DEFAULT_CONFIG.copy()
    
    def save_config(self, config: Dict[str, Any], profile_name: str) -> bool:
        """Sla een configuratieprofiel op.
        
        Args:
            config (Dict[str, Any]): De te bewaren configuratie.
            profile_name (str): Naam van het profiel.
            
        Returns:
            bool: True als succesvol opgeslagen, anders False.
        """
        self.ensure_config_dir()
        config_path = self.config_dir / f"{profile_name}.json"
        
        try:
            # Valideer de configuratie voordat deze wordt opgeslagen
            validated_config = validate_config(config)
            
            # Voeg metadata toe
            validated_config['_metadata'] = {
                'last_modified': datetime.datetime.now().isoformat(),
                'profile_name': profile_name
            }
            
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(validated_config, f, indent=2)
            return True
        except (IOError, TypeError, ValueError):
            return False
    
    def delete_config(self, profile_name: str) -> bool:
        """Verwijder een configuratieprofiel.
        
        Args:
            profile_name (str): Naam van het te verwijderen profiel.
            
        Returns:
            bool: True als succesvol verwijderd, anders False.
        """
        config_path = self.config_dir / f"{profile_name}.json"
        if not config_path.exists():
            return False
        
        try:
            config_path.unlink()
            return True
        except IOError:
            return False
    
    def export_config(self, profile_name: str, export_path: Path) -> bool:
        """Exporteer een configuratieprofiel naar een extern bestand.
        
        Args:
            profile_name (str): Naam van het te exporteren profiel.
            export_path (Path): Pad waar het profiel moet worden geëxporteerd.
            
        Returns:
            bool: True als succesvol geëxporteerd, anders False.
        """
        config = self.load_config(profile_name)
        if config == DEFAULT_CONFIG:
            return False
        
        try:
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2)
            return True
        except IOError:
            return False
    
    def import_config(self, import_path: Path, profile_name: Optional[str] = None) -> Tuple[bool, str]:
        """Importeer een configuratieprofiel uit een extern bestand.
        
        Args:
            import_path (Path): Pad naar het te importeren bestand.
            profile_name (Optional[str]): Naam voor het geïmporteerde profiel.
                Als None, wordt de bestandsnaam gebruikt.
                
        Returns:
            Tuple[bool, str]: (Succes, Profielnaam)
        """
        if not import_path.exists():
            return False, ""
        
        try:
            with open(import_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Valideer de configuratie
            config = validate_config(config)
            
            # Bepaal de profielnaam
            if profile_name is None:
                profile_name = import_path.stem
            
            # Sla de configuratie op
            success = self.save_config(config, profile_name)
            return success, profile_name
        except (json.JSONDecodeError, IOError):
            return False, ""


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Valideer en corrigeer een configuratieobject.
    
    Args:
        config (Dict[str, Any]): De te valideren configuratie.
        
    Returns:
        Dict[str, Any]: De gevalideerde en gecorrigeerde configuratie.
    """
    validated = DEFAULT_CONFIG.copy()
    
    # Valideer elke sectie en parameter
    for section, section_schema in CONFIG_SCHEMA.items():
        if section in config and isinstance(config[section], dict):
            for param, param_schema in section_schema.items():
                if param in config[section]:
                    value = config[section][param]
                    
                    # Valideer het type
                    if param_schema["type"] == "float":
                        try:
                            value = float(value)
                        except (ValueError, TypeError):
                            continue
                    
                    # Valideer de grenzen
                    if "min" in param_schema and value < param_schema["min"]:
                        value = param_schema["min"]
                    if "max" in param_schema and value > param_schema["max"]:
                        value = param_schema["max"]
                    
                    validated[section][param] = value
    
    return validated


def get_config_manager() -> ConfigManager:
    """Krijg een instantie van de ConfigManager.
    
    Returns:
        ConfigManager: Een instantie van de ConfigManager.
    """
    return ConfigManager()


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Laad configuratie uit een JSON-bestand of gebruik standaardwaarden.
    
    Args:
        config_path (Optional[str]): Pad naar het JSON-configuratiebestand.
            Als None, worden de standaardwaarden gebruikt.
            
    Returns:
        Dict[str, Any]: De geladen of standaard configuratie.
    """
    if config_path is None:
        return DEFAULT_CONFIG.copy()
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return validate_config(config)
    except (json.JSONDecodeError, IOError):
        return DEFAULT_CONFIG.copy()


def save_config(config: Dict[str, Any], config_path: str) -> bool:
    """Sla configuratie op naar een JSON-bestand.
    
    Args:
        config (Dict[str, Any]): De te bewaren configuratie.
        config_path (str): Pad waar het configuratiebestand moet worden opgeslagen.
        
    Returns:
        bool: True als succesvol opgeslagen, anders False.
    """
    try:
        # Zorg ervoor dat de map bestaat
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        # Valideer de configuratie voordat deze wordt opgeslagen
        validated_config = validate_config(config)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(validated_config, f, indent=2)
        return True
    except (IOError, TypeError):
        return False