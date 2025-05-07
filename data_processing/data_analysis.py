"""Module voor geavanceerde analyse van energiedata en visualisaties.

Deze module bevat functies voor:
- Het identificeren van patronen in energieproductie en -verbruik
- Het uitvoeren van statistische analyses op energiedata
- Het genereren van visualisaties voor inzicht in energiepatronen
- Het analyseren van potentiële besparingen door energieopslag
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, Optional, Union, List, Tuple
from datetime import datetime, timedelta


def identify_daily_patterns(data: pd.DataFrame) -> Dict[str, Any]:
    """Identificeer dagelijkse patronen in energieproductie en -verbruik.
    
    Args:
        data (pd.DataFrame): DataFrame met energiedata met een DatetimeIndex.
        
    Returns:
        Dict[str, Any]: Dictionary met geïdentificeerde dagelijkse patronen.
        
    Raises:
        ValueError: Als de data geen DatetimeIndex heeft.
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Data moet een DatetimeIndex hebben voor patroonanalyse")
    
    # Maak een kopie van de data met de uurwaarde als extra kolom
    df = data.copy()
    df['hour'] = df.index.hour
    
    # Bereken gemiddelde waarden per uur van de dag
    hourly_patterns = df.groupby('hour').mean()
    
    # Identificeer piekuren voor productie en verbruik
    production_peak_hour = hourly_patterns['Energy Produced (Wh)'].idxmax()
    consumption_peak_hour = hourly_patterns['Energy Consumed (Wh)'].idxmax()
    
    # Bereken surplus/tekort per uur
    hourly_patterns['Net Energy (Wh)'] = hourly_patterns['Energy Produced (Wh)'] - hourly_patterns['Energy Consumed (Wh)']
    
    # Identificeer uren met energieoverschot en -tekort
    surplus_hours = hourly_patterns[hourly_patterns['Net Energy (Wh)'] > 0].index.tolist()
    deficit_hours = hourly_patterns[hourly_patterns['Net Energy (Wh)'] < 0].index.tolist()
    
    # Bereken totale dagelijkse overschot en tekort
    total_daily_surplus = hourly_patterns[hourly_patterns['Net Energy (Wh)'] > 0]['Net Energy (Wh)'].sum()
    total_daily_deficit = abs(hourly_patterns[hourly_patterns['Net Energy (Wh)'] < 0]['Net Energy (Wh)'].sum())
    
    # Stel resultaten samen
    results = {
        'hourly_patterns': hourly_patterns,
        'production_peak_hour': production_peak_hour,
        'consumption_peak_hour': consumption_peak_hour,
        'surplus_hours': surplus_hours,
        'deficit_hours': deficit_hours,
        'total_daily_surplus': total_daily_surplus,
        'total_daily_deficit': total_daily_deficit,
        'storage_potential': min(total_daily_surplus, total_daily_deficit)
    }
    
    return results


def identify_weekly_patterns(data: pd.DataFrame) -> Dict[str, Any]:
    """Identificeer wekelijkse patronen in energieproductie en -verbruik.
    
    Args:
        data (pd.DataFrame): DataFrame met energiedata met een DatetimeIndex.
        
    Returns:
        Dict[str, Any]: Dictionary met geïdentificeerde wekelijkse patronen.
        
    Raises:
        ValueError: Als de data geen DatetimeIndex heeft.
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Data moet een DatetimeIndex hebben voor patroonanalyse")
    
    # Maak een kopie van de data met de dag van de week als extra kolom
    df = data.copy()
    df['day_of_week'] = df.index.dayofweek  # 0=maandag, 6=zondag
    
    # Bereken gemiddelde waarden per dag van de week
    daily_patterns = df.groupby('day_of_week').mean()
    
    # Identificeer dagen met hoogste productie en verbruik
    production_peak_day = daily_patterns['Energy Produced (Wh)'].idxmax()
    consumption_peak_day = daily_patterns['Energy Consumed (Wh)'].idxmax()
    
    # Bereken verschil tussen weekdagen en weekend
    weekday_mask = df['day_of_week'] < 5  # Maandag t/m vrijdag
    weekend_mask = df['day_of_week'] >= 5  # Zaterdag en zondag
    
    weekday_avg = df[weekday_mask].mean()
    weekend_avg = df[weekend_mask].mean()
    
    # Bereken netto energie per dag van de week
    daily_patterns['Net Energy (Wh)'] = daily_patterns['Energy Produced (Wh)'] - daily_patterns['Energy Consumed (Wh)']
    
    # Stel resultaten samen
    results = {
        'daily_patterns': daily_patterns,
        'production_peak_day': production_peak_day,
        'consumption_peak_day': consumption_peak_day,
        'weekday_avg': weekday_avg,
        'weekend_avg': weekend_avg,
        'weekday_weekend_diff': {
            'production': weekend_avg['Energy Produced (Wh)'] - weekday_avg['Energy Produced (Wh)'],
            'consumption': weekend_avg['Energy Consumed (Wh)'] - weekday_avg['Energy Consumed (Wh)']
        }
    }
    
    return results


def identify_seasonal_patterns(data: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Identificeer seizoensgebonden patronen in energiedata.
    
    Args:
        data (pd.DataFrame): DataFrame met energiedata met een DatetimeIndex.
        
    Returns:
        Optional[Dict[str, Any]]: Dictionary met seizoensgebonden patronen, of None als er
                                 onvoldoende data is voor seizoensanalyse.
        
    Raises:
        ValueError: Als de data geen DatetimeIndex heeft.
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Data moet een DatetimeIndex hebben voor seizoensanalyse")
    
    # Controleer of er voldoende data is voor seizoensanalyse (minimaal 2 maanden)
    unique_months = data.index.month.unique()
    if len(unique_months) < 2:
        return None
    
    # Maak een kopie van de data met de maand als extra kolom
    df = data.copy()
    df['month'] = df.index.month
    
    # Bereken gemiddelde waarden per maand
    monthly_patterns = df.groupby('month').mean()
    
    # Bepaal seizoenen (Noord-Halfrond)
    # Winter: 12, 1, 2; Lente: 3, 4, 5; Zomer: 6, 7, 8; Herfst: 9, 10, 11
    season_map = {
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Fall', 10: 'Fall', 11: 'Fall'
    }
    
    df['season'] = df['month'].map(season_map)
    
    # Bereken gemiddelde waarden per seizoen
    seasonal_patterns = df.groupby('season').mean()
    
    # Identificeer seizoenen met hoogste productie en verbruik
    if 'Energy Produced (Wh)' in seasonal_patterns.columns and len(seasonal_patterns) > 1:
        production_peak_season = seasonal_patterns['Energy Produced (Wh)'].idxmax()
        consumption_peak_season = seasonal_patterns['Energy Consumed (Wh)'].idxmax()
    else:
        production_peak_season = None
        consumption_peak_season = None
    
    # Stel resultaten samen
    results = {
        'monthly_patterns': monthly_patterns,
        'seasonal_patterns': seasonal_patterns,
        'production_peak_season': production_peak_season,
        'consumption_peak_season': consumption_peak_season
    }
    
    return results


def analyze_energy_balance(data: pd.DataFrame) -> Dict[str, Any]:
    """Analyseer de energiebalans en identificeer potentiële opslagmogelijkheden.
    
    Args:
        data (pd.DataFrame): DataFrame met energiedata.
        
    Returns:
        Dict[str, Any]: Dictionary met energiebalans analyses.
    """
    # Bereken netto energie als die nog niet bestaat
    if 'Net Energy (Wh)' not in data.columns:
        data['Net Energy (Wh)'] = data['Energy Produced (Wh)'] - data['Energy Consumed (Wh)']
    
    # Bereken totale overschotten en tekorten
    total_surplus = data[data['Net Energy (Wh)'] > 0]['Net Energy (Wh)'].sum()
    total_deficit = abs(data[data['Net Energy (Wh)'] < 0]['Net Energy (Wh)'].sum())
    
    # Bereken percentage van de tijd met overschot/tekort
    surplus_percentage = (data['Net Energy (Wh)'] > 0).mean() * 100
    deficit_percentage = (data['Net Energy (Wh)'] < 0).mean() * 100
    
    # Bereken maximale theoretische opslagcapaciteit nodig
    # (cumulatieve som van netto energie, maximum verschil tussen hoogste en laagste punt)
    cumulative_energy = data['Net Energy (Wh)'].cumsum()
    max_storage_needed = cumulative_energy.max() - cumulative_energy.min()
    
    # Bereken potentiële zelfvoorzienendheid met perfecte opslag
    total_consumption = data['Energy Consumed (Wh)'].sum()
    total_production = data['Energy Produced (Wh)'].sum()
    
    current_self_sufficiency = 100 - (data['Imported from Grid (Wh)'].sum() / total_consumption * 100)
    theoretical_max_self_sufficiency = min(100, (total_production / total_consumption * 100))
    
    # Bereken potentiële verbetering door opslag
    storage_improvement_potential = theoretical_max_self_sufficiency - current_self_sufficiency
    
    # Stel resultaten samen
    results = {
        'total_surplus': total_surplus,
        'total_deficit': total_deficit,
        'surplus_percentage': surplus_percentage,
        'deficit_percentage': deficit_percentage,
        'max_storage_needed': max_storage_needed,
        'current_self_sufficiency': current_self_sufficiency,
        'theoretical_max_self_sufficiency': theoretical_max_self_sufficiency,
        'storage_improvement_potential': storage_improvement_potential
    }
    
    return results


def detect_outliers(data: pd.DataFrame, columns: List[str] = None, method: str = 'iqr', threshold: float = 1.5) -> Dict[str, pd.DataFrame]:
    """Detecteer uitschieters in de energiedata.
    
    Args:
        data (pd.DataFrame): DataFrame met energiedata.
        columns (List[str], optional): Lijst met kolomnamen om te analyseren. Standaard None (alle numerieke kolommen).
        method (str, optional): Methode voor outlier detectie ('iqr' of 'zscore'). Standaard 'iqr'.
        threshold (float, optional): Drempelwaarde voor outlier detectie. Standaard 1.5 voor IQR, 3 voor z-score.
        
    Returns:
        Dict[str, pd.DataFrame]: Dictionary met DataFrames van gedetecteerde outliers per kolom.
        
    Raises:
        ValueError: Als een ongeldige methode wordt opgegeven.
    """
    if columns is None:
        # Selecteer alleen numerieke kolommen
        columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    outliers = {}
    
    for col in columns:
        if col not in data.columns:
            continue
            
        if method.lower() == 'iqr':
            # IQR methode
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            outliers[col] = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
            
        elif method.lower() == 'zscore':
            # Z-score methode
            mean = data[col].mean()
            std = data[col].std()
            
            if std == 0:  # Voorkom delen door nul
                outliers[col] = pd.DataFrame()
                continue
                
            z_scores = abs((data[col] - mean) / std)
            outliers[col] = data[z_scores > threshold]
            
        else:
            raise ValueError(f"Ongeldige methode: {method}. Gebruik 'iqr' of 'zscore'.")
    
    return outliers


def create_time_series_plot(data: pd.DataFrame, columns: List[str], title: str = "Energiedata Tijdreeks", 
                           resample: Optional[str] = None) -> go.Figure:
    """Creëer een tijdreeksplot van energiedata.
    
    Args:
        data (pd.DataFrame): DataFrame met energiedata met een DatetimeIndex.
        columns (List[str]): Lijst met kolomnamen om te plotten.
        title (str, optional): Titel van de plot. Standaard "Energiedata Tijdreeks".
        resample (Optional[str], optional): Optioneel resampling interval (bijv. 'D' voor dagelijks). Standaard None.
        
    Returns:
        go.Figure: Plotly figuur object met de tijdreeksplot.
        
    Raises:
        ValueError: Als de data geen DatetimeIndex heeft.
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Data moet een DatetimeIndex hebben voor tijdreeksplots")
    
    # Maak een kopie van de data en resample indien nodig
    plot_data = data.copy()
    if resample is not None:
        plot_data = plot_data.resample(resample).mean()
    
    # Creëer de figuur
    fig = go.Figure()
    
    # Voeg elke kolom toe als een aparte lijn
    for column in columns:
        if column in plot_data.columns:
            fig.add_trace(go.Scatter(
                x=plot_data.index,
                y=plot_data[column],
                mode='lines',
                name=column
            ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Datum/Tijd",
        yaxis_title="Energie (Wh)",
        legend_title="Variabelen",
        hovermode="x unified"
    )
    
    return fig


def create_energy_balance_heatmap(data: pd.DataFrame) -> go.Figure:
    """Creëer een heatmap van de energiebalans per uur en dag.
    
    Args:
        data (pd.DataFrame): DataFrame met energiedata met een DatetimeIndex.
        
    Returns:
        go.Figure: Plotly figuur object met de heatmap.
        
    Raises:
        ValueError: Als de data geen DatetimeIndex heeft.
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Data moet een DatetimeIndex hebben voor heatmap creatie")
    
    # Bereken netto energie als die nog niet bestaat
    if 'Net Energy (Wh)' not in data.columns:
        data['Net Energy (Wh)'] = data['Energy Produced (Wh)'] - data['Energy Consumed (Wh)']
    
    # Maak een pivot tabel met uren op de y-as en dagen op de x-as
    pivot_data = data.copy()
    pivot_data['hour'] = pivot_data.index.hour
    pivot_data['date'] = pivot_data.index.date
    
    # Creëer de pivot tabel
    heatmap_data = pivot_data.pivot_table(
        values='Net Energy (Wh)', 
        index='hour',
        columns='date',
        aggfunc='mean'
    )
    
    # Creëer de heatmap
    fig = px.imshow(
        heatmap_data,
        labels=dict(x="Datum", y="Uur van de dag", color="Netto Energie (Wh)"),
        x=heatmap_data.columns,
        y=heatmap_data.index,
        color_continuous_scale="RdBu_r",  # Rood voor negatief, blauw voor positief
        origin="lower"
    )
    
    fig.update_layout(
        title="Energiebalans Heatmap (Blauw = Overschot, Rood = Tekort)",
        xaxis_title="Datum",
        yaxis_title="Uur van de dag"
    )
    
    return fig


def create_daily_profile_plot(data: pd.DataFrame, group_by: str = 'all') -> go.Figure:
    """Creëer een plot van het gemiddelde dagelijkse energieprofiel.
    
    Args:
        data (pd.DataFrame): DataFrame met energiedata met een DatetimeIndex.
        group_by (str, optional): Hoe de data te groeperen ('all', 'weekday_weekend', 'season'). Standaard 'all'.
        
    Returns:
        go.Figure: Plotly figuur object met het dagelijkse profiel.
        
    Raises:
        ValueError: Als de data geen DatetimeIndex heeft of als een ongeldige group_by waarde wordt opgegeven.
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Data moet een DatetimeIndex hebben voor profielanalyse")
    
    valid_group_by = ['all', 'weekday_weekend', 'season']
    if group_by not in valid_group_by:
        raise ValueError(f"Ongeldige group_by waarde: {group_by}. Geldige waarden zijn: {valid_group_by}")
    
    # Maak een kopie van de data met de uurwaarde als extra kolom
    df = data.copy()
    df['hour'] = df.index.hour
    
    # Creëer de figuur
    fig = go.Figure()
    
    if group_by == 'all':
        # Bereken gemiddelde waarden per uur voor de hele dataset
        hourly_avg = df.groupby('hour').mean()
        
        # Voeg lijnen toe voor productie en verbruik
        fig.add_trace(go.Scatter(
            x=hourly_avg.index,
            y=hourly_avg['Energy Produced (Wh)'],
            mode='lines',
            name='Productie'
        ))
        
        fig.add_trace(go.Scatter(
            x=hourly_avg.index,
            y=hourly_avg['Energy Consumed (Wh)'],
            mode='lines',
            name='Verbruik'
        ))
        
        title = "Gemiddeld Dagelijks Energieprofiel"
        
    elif group_by == 'weekday_weekend':
        # Voeg dag van de week toe
        df['is_weekend'] = df.index.dayofweek >= 5  # 5=zaterdag, 6=zondag
        
        # Bereken gemiddelde waarden per uur voor weekdagen en weekenden
        weekday_avg = df[~df['is_weekend']].groupby('hour').mean()
        weekend_avg = df[df['is_weekend']].groupby('hour').mean()
        
        # Voeg lijnen toe voor weekdagen
        fig.add_trace(go.Scatter(
            x=weekday_avg.index,
            y=weekday_avg['Energy Produced (Wh)'],
            mode='lines',
            name='Productie (Weekdag)'
        ))
        
        fig.add_trace(go.Scatter(
            x=weekday_avg.index,
            y=weekday_avg['Energy Consumed (Wh)'],
            mode='lines',
            name='Verbruik (Weekdag)'
        ))
        
        # Voeg lijnen toe voor weekenden
        fig.add_trace(go.Scatter(
            x=weekend_avg.index,
            y=weekend_avg['Energy Produced (Wh)'],
            mode='lines',
            name='Productie (Weekend)'
        ))
        
        fig.add_trace(go.Scatter(
            x=weekend_avg.index,
            y=weekend_avg['Energy Consumed (Wh)'],
            mode='lines',
            name='Verbruik (Weekend)'
        ))
        
        title = "Gemiddeld Dagelijks Energieprofiel (Weekdag vs. Weekend)"
        
    elif group_by == 'season':
        # Controleer of er voldoende data is voor seizoensanalyse
        if len(df.index.month.unique()) < 2:
            raise ValueError("Onvoldoende data voor seizoensanalyse (minimaal 2 maanden nodig)")
        
        # Voeg seizoen toe
        season_map = {
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        }
        
        df['month'] = df.index.month
        df['season'] = df['month'].map(season_map)
        
        # Bereken gemiddelde waarden per uur per seizoen
        for season in df['season'].unique():
            season_data = df[df['season'] == season]
            season_avg = season_data.groupby('hour').mean()
            
            # Voeg lijnen toe voor dit seizoen
            fig.add_trace(go.Scatter(
                x=season_avg.index,
                y=season_avg['Energy Produced (Wh)'],
                mode='lines',
                name=f'Productie ({season})'
            ))
            
            fig.add_trace(go.Scatter(
                x=season_avg.index,
                y=season_avg['Energy Consumed (Wh)'],
                mode='lines',
                name=f'Verbruik ({season})'
            ))
        
        title = "Gemiddeld Dagelijks Energieprofiel per Seizoen"
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Uur van de dag",
        yaxis_title="Energie (Wh)",
        legend_title="Variabelen",
        hovermode="x unified",
        xaxis=dict(tickmode='linear', tick0=0, dtick=1)
    )
    
    return fig


def calculate_storage_potential(data: pd.DataFrame, capacity: float = float('inf')) -> Dict[str, Any]:
    """Bereken het potentieel voor energieopslag op basis van de energiedata.
    
    Args:
        data (pd.DataFrame): DataFrame met energiedata.
        capacity (float, optional): Maximale opslagcapaciteit in Wh. Standaard oneindig.
        
    Returns:
        Dict[str, Any]: Dictionary met opslagpotentieel analyses.
    """
    # Bereken netto energie als die nog niet bestaat
    if 'Net Energy (Wh)' not in data.columns:
        data['Net Energy (Wh)'] = data['Energy Produced (Wh)'] - data['Energy Consumed (Wh)']
    
    # Maak een kopie van de data voor simulatie
    df = data.copy()
    
    # Initialiseer opslagvariabelen
    df['Storage Level (Wh)'] = 0.0
    df['Energy Stored (Wh)'] = 0.0
    df['Energy Used from Storage (Wh)'] = 0.0
    df['Storage Overflow (Wh)'] = 0.0
    
    # Simuleer opslag voor elke tijdstap
    storage_level = 0.0
    
    for idx in df.index:
        net_energy = df.at[idx, 'Net Energy (Wh)']
        
        if net_energy > 0:  # Overschot, opslaan
            energy_to_store = net_energy
            available_capacity = capacity - storage_level
            
            if energy_to_store <= available_capacity:
                storage_level += energy_to_store
                df.at[idx, 'Energy Stored (Wh)'] = energy_to_store
                df.at[idx, 'Storage Overflow (Wh)'] = 0
            else:
                df.at[idx, 'Energy Stored (Wh)'] = available_capacity
                df.at[idx, 'Storage Overflow (Wh)'] = energy_to_store - available_capacity
                storage_level = capacity
        
        else:  # Tekort, gebruiken uit opslag
            energy_needed = abs(net_energy)
            
            if energy_needed <= storage_level:
                storage_level -= energy_needed
                df.at[idx, 'Energy Used from Storage (Wh)'] = energy_needed
            else:
                df.at[idx, 'Energy Used from Storage (Wh)'] = storage_level
                storage_level = 0
        
        df.at[idx, 'Storage Level (Wh)'] = storage_level
    
    # Bereken statistieken
    total_stored = df['Energy Stored (Wh)'].sum()
    total_used = df['Energy Used from Storage (Wh)'].sum()
    total_overflow = df['Storage Overflow (Wh)'].sum()
    storage_efficiency = (total_used / total_stored * 100) if total_stored > 0 else 0
    
    # Bereken maximaal bereikte opslaglevel
    max_storage_level = df['Storage Level (Wh)'].max()
    
    # Bereken verbeterde zelfvoorzienendheid
    original_import = df['Imported from Grid (Wh)'].sum()
    reduced_import = original_import - total_used
    total_consumption = df['Energy Consumed (Wh)'].sum()
    
    original_self_sufficiency = 100 - (original_import / total_consumption * 100)
    improved_self_sufficiency = 100 - (reduced_import / total_consumption * 100)
    
    # Stel resultaten samen
    results = {
        'simulation_data': df,
        'total_energy_stored': total_stored,
        'total_energy_used_from_storage': total_used,
        'total_storage_overflow': total_overflow,
        'storage_efficiency': storage_efficiency,
        'max_storage_level': max_storage_level,
        'original_self_sufficiency': original_self_sufficiency,
        'improved_self_sufficiency': improved_self_sufficiency,
        'self_sufficiency_improvement': improved_self_sufficiency - original_self_sufficiency
    }
    
    return results