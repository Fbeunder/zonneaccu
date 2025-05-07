"""Module voor het laden, valideren en voorbereiden van energiedata.

Deze module bevat functies voor:
- Het inlezen van CSV-data met energieproductie en -verbruik
- Het valideren van de ingelezen data
- Het converteren van de data naar geschikte formaten voor verdere analyse
- Het genereren van basisstatistieken over de ingelezen data
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union, List, Tuple
from datetime import datetime


def load_csv_data(file_path: str) -> pd.DataFrame:
    """Laad energiedata uit een CSV-bestand.
    
    Args:
        file_path (str): Pad naar het CSV-bestand met energiedata.
        
    Returns:
        pd.DataFrame: DataFrame met de geladen data.
        
    Raises:
        FileNotFoundError: Als het bestand niet gevonden kan worden.
        ValueError: Als het bestand geen geldige energiedata bevat.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Bestand niet gevonden: {file_path}")
    
    try:
        # Probeer het bestand te laden met verschillende opties voor datetime parsing
        try:
            # Eerst proberen met standaard Europees datumformaat (DD/MM/YYYY)
            df = pd.read_csv(file_path, parse_dates=['Date/Time'], dayfirst=True)
        except:
            try:
                # Als dat niet lukt, probeer Amerikaans formaat (MM/DD/YYYY)
                df = pd.read_csv(file_path, parse_dates=['Date/Time'], dayfirst=False)
            except:
                # Als laatste optie, laad zonder parsing en converteer daarna
                df = pd.read_csv(file_path)
                df['Date/Time'] = pd.to_datetime(df['Date/Time'], errors='coerce')
        
        # Controleer of de datetime kolom correct is geparsed
        if df['Date/Time'].isna().any():
            raise ValueError("Sommige datumwaarden konden niet worden geparsed")
        
        # Zet de Date/Time kolom als index
        df.set_index('Date/Time', inplace=True)
        
        return df
    
    except Exception as e:
        raise ValueError(f"Fout bij het laden van het CSV-bestand: {str(e)}")


def validate_data(data: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Valideer de geladen energiedata.
    
    Args:
        data (pd.DataFrame): De te valideren data.
        
    Returns:
        Tuple[bool, List[str]]: Een tuple met een boolean die aangeeft of de data
                               geldig is, en een lijst met eventuele foutberichten.
    """
    errors = []
    
    # Controleer of de vereiste kolommen aanwezig zijn
    required_columns = [
        'Energy Produced (Wh)', 
        'Energy Consumed (Wh)', 
        'Exported to Grid (Wh)', 
        'Imported from Grid (Wh)'
    ]
    
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        errors.append(f"Ontbrekende kolommen: {', '.join(missing_columns)}")
    
    # Als er kolommen ontbreken, kunnen we niet verder valideren
    if errors:
        return False, errors
    
    # Controleer op negatieve waarden (energiewaarden moeten positief zijn)
    for col in required_columns:
        if (data[col] < 0).any():
            errors.append(f"Negatieve waarden gevonden in kolom {col}")
    
    # Controleer of de data consistent is (geproduceerd + geïmporteerd = verbruikt + geëxporteerd)
    tolerance = 1.0  # Kleine tolerantie voor afrondingsfouten
    energy_balance = data['Energy Produced (Wh)'] + data['Imported from Grid (Wh)'] - \
                     data['Energy Consumed (Wh)'] - data['Exported to Grid (Wh)']
    
    if (abs(energy_balance) > tolerance).any():
        errors.append("Energiebalans klopt niet voor sommige tijdstippen")
    
    # Controleer op ontbrekende waarden
    if data.isna().any().any():
        errors.append("Dataset bevat ontbrekende waarden")
    
    return len(errors) == 0, errors


def resample_data(data: pd.DataFrame, interval: str = '15min') -> pd.DataFrame:
    """Hersampling van de data naar een specifiek tijdsinterval.
    
    Args:
        data (pd.DataFrame): De energiedata.
        interval (str): Het gewenste tijdsinterval (bijv. '15min', '1H', '1D').
        
    Returns:
        pd.DataFrame: De geresamplede data.
    """
    # Controleer of de index een DatetimeIndex is
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Data moet een DatetimeIndex hebben om te kunnen resampling")
    
    # Resample de data naar het gewenste interval
    resampled = data.resample(interval).mean()
    
    # Vul eventuele ontbrekende waarden in
    resampled.fillna(method='ffill', inplace=True)
    
    return resampled


def calculate_derived_values(data: pd.DataFrame) -> pd.DataFrame:
    """Bereken afgeleide waarden uit de energiedata.
    
    Args:
        data (pd.DataFrame): De energiedata.
        
    Returns:
        pd.DataFrame: De data met toegevoegde afgeleide kolommen.
    """
    # Maak een kopie om de originele data niet te wijzigen
    result = data.copy()
    
    # Bereken netto energieoverschot/tekort
    result['Net Energy (Wh)'] = result['Energy Produced (Wh)'] - result['Energy Consumed (Wh)']
    
    # Bereken zelfconsumptie (hoeveel van de geproduceerde energie direct wordt verbruikt)
    result['Self Consumption (Wh)'] = result['Energy Produced (Wh)'] - result['Exported to Grid (Wh)']
    
    # Bereken zelfvoorzienendheid (percentage van verbruik dat door eigen productie wordt gedekt)
    with np.errstate(divide='ignore', invalid='ignore'):
        result['Self Sufficiency (%)'] = (result['Self Consumption (Wh)'] / result['Energy Consumed (Wh)']) * 100
    
    # Vervang oneindige waarden door NaN en vul NaN-waarden in met 0
    result.replace([np.inf, -np.inf], np.nan, inplace=True)
    result.fillna(0, inplace=True)
    
    return result


def calculate_basic_statistics(data: pd.DataFrame) -> Dict[str, Any]:
    """Bereken basisstatistieken van de energiedata.
    
    Args:
        data (pd.DataFrame): De energiedata.
        
    Returns:
        Dict[str, Any]: Een dictionary met basisstatistieken.
    """
    stats = {}
    
    # Controleer of de vereiste kolommen aanwezig zijn
    energy_columns = [
        'Energy Produced (Wh)', 
        'Energy Consumed (Wh)', 
        'Exported to Grid (Wh)', 
        'Imported from Grid (Wh)'
    ]
    
    # Bereken alleen statistieken voor kolommen die bestaan
    available_columns = [col for col in energy_columns if col in data.columns]
    
    # Totalen berekenen
    for col in available_columns:
        stats[f'Total {col}'] = data[col].sum()
    
    # Als we afgeleide waarden hebben, bereken ook daarvoor statistieken
    if 'Net Energy (Wh)' in data.columns:
        stats['Total Net Energy (Wh)'] = data['Net Energy (Wh)'].sum()
        stats['Positive Net Energy Hours'] = (data['Net Energy (Wh)'] > 0).sum()
        stats['Negative Net Energy Hours'] = (data['Net Energy (Wh)'] < 0).sum()
    
    # Bereken gemiddelden per dag, indien mogelijk
    if isinstance(data.index, pd.DatetimeIndex):
        # Aantal unieke dagen in de dataset
        unique_days = len(data.index.date.unique())
        if unique_days > 0:
            for col in available_columns:
                stats[f'Average Daily {col}'] = stats[f'Total {col}'] / unique_days
    
    # Bereken pieken
    for col in available_columns:
        stats[f'Peak {col}'] = data[col].max()
        stats[f'Peak {col} Time'] = data[col].idxmax()
    
    # Bereken zelfvoorzienendheid als percentage
    if 'Energy Produced (Wh)' in data.columns and 'Energy Consumed (Wh)' in data.columns:
        total_produced = data['Energy Produced (Wh)'].sum()
        total_consumed = data['Energy Consumed (Wh)'].sum()
        if total_consumed > 0:
            stats['Overall Self Sufficiency (%)'] = min(100, (total_produced / total_consumed) * 100)
        else:
            stats['Overall Self Sufficiency (%)'] = 0
    
    return stats


def create_sample_data(output_path: str, days: int = 1) -> str:
    """Creëer een voorbeelddataset met energiedata.
    
    Args:
        output_path (str): Pad waar de voorbeelddata moet worden opgeslagen.
        days (int): Aantal dagen aan data om te genereren.
        
    Returns:
        str: Pad naar het gegenereerde bestand.
    """
    # Maak een tijdsindex voor het aantal opgegeven dagen met 15-minuten intervallen
    start_date = '2024-01-01'
    periods = days * 24 * 4  # 4 kwartieren per uur, 24 uur per dag
    date_range = pd.date_range(start=start_date, periods=periods, freq='15min')
    
    # Maak een lege DataFrame met de tijdsindex
    df = pd.DataFrame(index=date_range)
    
    # Genereer realistische energieproductiedata (alleen overdag productie)
    hour_of_day = df.index.hour
    is_daytime = (hour_of_day >= 7) & (hour_of_day <= 19)  # Tussen 7:00 en 19:00
    
    # Basispatroon voor zonne-energieproductie (piek rond het middaguur)
    solar_base = np.zeros(len(df))
    solar_base[is_daytime] = np.sin(np.pi * (hour_of_day[is_daytime] - 7) / 12) * 500
    
    # Voeg wat ruis toe voor realisme
    np.random.seed(42)  # Voor reproduceerbaarheid
    noise = np.random.normal(0, 20, len(df))
    
    # Energieproductie (in Wh)
    df['Energy Produced (Wh)'] = np.maximum(0, solar_base + noise)
    
    # Energieverbruik (basispatroon met pieken 's ochtends en 's avonds)
    base_consumption = 50 + 20 * np.sin(np.pi * hour_of_day / 12) + 150 * (hour_of_day >= 17) * (hour_of_day <= 22)
    df['Energy Consumed (Wh)'] = base_consumption + np.random.normal(0, 10, len(df))
    
    # Bereken export en import
    energy_balance = df['Energy Produced (Wh)'] - df['Energy Consumed (Wh)']
    df['Exported to Grid (Wh)'] = np.maximum(0, energy_balance)
    df['Imported from Grid (Wh)'] = np.maximum(0, -energy_balance)
    
    # Rond alle waarden af op hele getallen voor leesbaarheid
    for col in df.columns:
        df[col] = df[col].round(0)
    
    # Reset de index om de datum/tijd als kolom te krijgen
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'Date/Time'}, inplace=True)
    
    # Sla op als CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    return output_path