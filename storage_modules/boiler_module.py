"""Module voor berekeningen gerelateerd aan warmwaterboiler energieopslag.

Deze module bevat functies voor:
- Het modelleren van een warmwaterboiler als energieopslagmedium
- Het berekenen van energieopslagcapaciteit en warmteverliezen
- Het simuleren van energieopslag in een warmwaterboiler
- Het analyseren van besparingen door gebruik van een warmwaterboiler
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, Any, Optional, Union, List, Tuple
from datetime import datetime, timedelta

# Constanten voor thermodynamische berekeningen
WATER_SPECIFIC_HEAT = 4186  # J/(kg·K) - Specifieke warmtecapaciteit van water
WATER_DENSITY = 1000  # kg/m³ - Dichtheid van water
J_TO_WH = 1 / 3600  # Conversie van Joule naar Watt-uur


class BoilerModel:
    """Model van een warmwaterboiler voor energieopslag.
    
    Deze klasse bevat de eigenschappen en methoden voor het modelleren van een
    warmwaterboiler als energieopslagmedium, inclusief berekeningen voor
    energieopslagcapaciteit, warmteverliezen en efficiëntie.
    """
    
    def __init__(self, 
                 volume: float = 200,  # liter
                 min_temp: float = 40,  # °C
                 max_temp: float = 85,  # °C
                 ambient_temp: float = 20,  # °C
                 heat_loss_coefficient: float = 1.5,  # W/(m²·K)
                 surface_area: Optional[float] = None,  # m²
                 heating_power: float = 2000,  # W
                 efficiency: float = 0.98,  # 98%
                 gas_price: float = 1.20,  # €/m³
                 electricity_price: float = 0.30,  # €/kWh
                 gas_efficiency: float = 0.90,  # 90%
                 gas_energy_density: float = 9.77  # kWh/m³
                ):
        """Initialiseer een BoilerModel met de gegeven parameters.
        
        Args:
            volume: Boilervolume in liters
            min_temp: Minimale watertemperatuur in °C
            max_temp: Maximale watertemperatuur in °C
            ambient_temp: Omgevingstemperatuur in °C
            heat_loss_coefficient: Warmteverliescoëfficiënt in W/(m²·K)
            surface_area: Oppervlakte van de boiler in m² (indien None, wordt berekend op basis van volume)
            heating_power: Verwarmingsvermogen in W
            efficiency: Efficiëntie van elektrische verwarming (0-1)
            gas_price: Gasprijs in €/m³
            electricity_price: Elektriciteitsprijs in €/kWh
            gas_efficiency: Efficiëntie van gasverwarming (0-1)
            gas_energy_density: Energiedichtheid van gas in kWh/m³
        """
        self.volume = volume  # liter
        self.min_temp = min_temp  # °C
        self.max_temp = max_temp  # °C
        self.ambient_temp = ambient_temp  # °C
        self.heat_loss_coefficient = heat_loss_coefficient  # W/(m²·K)
        self.heating_power = heating_power  # W
        self.efficiency = efficiency  # fractie (0-1)
        self.gas_price = gas_price  # €/m³
        self.electricity_price = electricity_price  # €/kWh
        self.gas_efficiency = gas_efficiency  # fractie (0-1)
        self.gas_energy_density = gas_energy_density  # kWh/m³
        
        # Bereken oppervlakte als die niet is opgegeven
        if surface_area is None:
            # Schat oppervlakte op basis van volume (aanname: cilinder met hoogte = 2 * diameter)
            radius = (volume / (2 * np.pi * 1000)) ** (1/3)  # m
            height = 2 * 2 * radius  # m
            self.surface_area = 2 * np.pi * radius * (radius + height)  # m²
        else:
            self.surface_area = surface_area  # m²
    
    def calculate_storage_capacity(self) -> float:
        """Bereken de maximale energieopslagcapaciteit van de boiler in Wh.
        
        Returns:
            float: Opslagcapaciteit in Wh
        """
        # Bereken de energie nodig om het water van min_temp naar max_temp te verwarmen
        mass = self.volume * WATER_DENSITY / 1000  # kg (volume in liter)
        temp_diff = self.max_temp - self.min_temp  # °C of K
        energy_joules = mass * WATER_SPECIFIC_HEAT * temp_diff  # J
        energy_wh = energy_joules * J_TO_WH  # Wh
        
        return energy_wh
    
    def calculate_heat_loss(self, current_temp: float, duration_hours: float) -> float:
        """Bereken het warmteverlies over een bepaalde tijdsperiode.
        
        Args:
            current_temp: Huidige watertemperatuur in °C
            duration_hours: Tijdsduur in uren
            
        Returns:
            float: Warmteverlies in Wh
        """
        temp_diff = current_temp - self.ambient_temp  # °C of K
        heat_loss_rate = self.heat_loss_coefficient * self.surface_area * temp_diff  # W
        heat_loss_wh = heat_loss_rate * duration_hours  # Wh
        
        return heat_loss_wh
    
    def calculate_temperature_drop(self, current_temp: float, duration_hours: float) -> float:
        """Bereken de temperatuurdaling over een bepaalde tijdsperiode.
        
        Args:
            current_temp: Huidige watertemperatuur in °C
            duration_hours: Tijdsduur in uren
            
        Returns:
            float: Nieuwe temperatuur in °C na de gegeven tijdsduur
        """
        heat_loss_wh = self.calculate_heat_loss(current_temp, duration_hours)
        mass = self.volume * WATER_DENSITY / 1000  # kg
        
        # Bereken temperatuurdaling
        temp_drop = heat_loss_wh / (mass * WATER_SPECIFIC_HEAT * J_TO_WH)  # °C
        new_temp = current_temp - temp_drop
        
        # Temperatuur kan niet lager worden dan omgevingstemperatuur
        return max(new_temp, self.ambient_temp)
    
    def calculate_heating_time(self, start_temp: float, target_temp: float) -> float:
        """Bereken de tijd nodig om water van start_temp naar target_temp te verwarmen.
        
        Args:
            start_temp: Starttemperatuur in °C
            target_temp: Doeltemperatuur in °C
            
        Returns:
            float: Verwarmingstijd in uren
        """
        if start_temp >= target_temp:
            return 0.0
        
        mass = self.volume * WATER_DENSITY / 1000  # kg
        temp_diff = target_temp - start_temp  # °C
        energy_joules = mass * WATER_SPECIFIC_HEAT * temp_diff  # J
        energy_wh = energy_joules * J_TO_WH  # Wh
        
        # Tijd = energie / (vermogen * efficiëntie)
        heating_time_hours = energy_wh / (self.heating_power * self.efficiency / 1000)  # uren
        
        return heating_time_hours
    
    def calculate_gas_cost(self, energy_wh: float) -> float:
        """Bereken de kosten voor het verwarmen met gas.
        
        Args:
            energy_wh: Benodigde energie in Wh
            
        Returns:
            float: Kosten in €
        """
        energy_kwh = energy_wh / 1000  # kWh
        gas_m3 = energy_kwh / (self.gas_energy_density * self.gas_efficiency)  # m³
        cost = gas_m3 * self.gas_price  # €
        
        return cost
    
    def calculate_electricity_cost(self, energy_wh: float) -> float:
        """Bereken de kosten voor het verwarmen met elektriciteit.
        
        Args:
            energy_wh: Benodigde energie in Wh
            
        Returns:
            float: Kosten in €
        """
        energy_kwh = energy_wh / 1000  # kWh
        cost = energy_kwh * self.electricity_price / self.efficiency  # €
        
        return cost
    
    def calculate_savings(self, energy_wh: float) -> float:
        """Bereken de besparingen door elektriciteit i.p.v. gas te gebruiken.
        
        Args:
            energy_wh: Opgeslagen energie in Wh
            
        Returns:
            float: Besparingen in €
        """
        gas_cost = self.calculate_gas_cost(energy_wh)
        electricity_cost = self.calculate_electricity_cost(energy_wh)
        
        return gas_cost - electricity_cost


def simulate_boiler_storage(data: pd.DataFrame, 
                           boiler: BoilerModel,
                           usage_profile: Dict[int, float] = None,
                           initial_temp: float = None) -> Dict[str, Any]:
    """Simuleer energieopslag in een warmwaterboiler op basis van energiedata.
    
    Args:
        data: DataFrame met energiedata met een DatetimeIndex
        boiler: BoilerModel object met boilereigenschappen
        usage_profile: Dictionary met uren (0-23) als keys en waterverbruik in liters als values
        initial_temp: Initiële watertemperatuur in °C (standaard: boiler.min_temp)
        
    Returns:
        Dict[str, Any]: Resultaten van de simulatie
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Data moet een DatetimeIndex hebben voor boilersimulatie")
    
    # Controleer of de benodigde kolommen aanwezig zijn
    required_columns = ['Net Energy (Wh)', 'Energy Produced (Wh)', 'Energy Consumed (Wh)']
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Kolom '{col}' ontbreekt in de data")
    
    # Standaard gebruiksprofiel als er geen is opgegeven
    if usage_profile is None:
        # Standaard: ochtend- en avondpiek in warmwatergebruik
        usage_profile = {
            7: 50,   # 50 liter om 7:00
            8: 30,   # 30 liter om 8:00
            19: 40,  # 40 liter om 19:00
            22: 30   # 30 liter om 22:00
        }
    
    # Initiële temperatuur
    if initial_temp is None:
        initial_temp = boiler.min_temp
    
    # Maak een kopie van de data voor simulatie
    df = data.copy()
    
    # Voeg kolommen toe voor boilersimulatie
    df['Boiler Temperature (°C)'] = 0.0
    df['Boiler Energy Content (Wh)'] = 0.0
    df['Energy to Boiler (Wh)'] = 0.0
    df['Energy from Boiler (Wh)'] = 0.0
    df['Heat Loss (Wh)'] = 0.0
    df['Hot Water Usage (L)'] = 0.0
    
    # Bereken tijdstap in uren
    if len(df) > 1:
        time_step_hours = (df.index[1] - df.index[0]).total_seconds() / 3600
    else:
        time_step_hours = 1.0  # Standaard 1 uur als er maar één rij is
    
    # Bereken maximale energieinhoud van de boiler
    max_energy_content = boiler.calculate_storage_capacity()
    
    # Initialiseer simulatievariabelen
    current_temp = initial_temp
    current_energy_content = (current_temp - boiler.min_temp) / (boiler.max_temp - boiler.min_temp) * max_energy_content
    
    # Simuleer voor elke tijdstap
    for i, idx in enumerate(df.index):
        # Sla huidige status op
        df.at[idx, 'Boiler Temperature (°C)'] = current_temp
        df.at[idx, 'Boiler Energy Content (Wh)'] = current_energy_content
        
        # Bepaal warmwatergebruik op basis van het gebruiksprofiel
        hour = idx.hour
        water_usage = usage_profile.get(hour, 0.0)  # liter
        df.at[idx, 'Hot Water Usage (L)'] = water_usage
        
        # Bereken energieverlies door warmwatergebruik
        if water_usage > 0:
            # Energie nodig om koud water (10°C) op te warmen naar huidige temperatuur
            energy_used = water_usage * WATER_DENSITY / 1000 * WATER_SPECIFIC_HEAT * (current_temp - 10) * J_TO_WH  # Wh
            energy_used = min(energy_used, current_energy_content)  # Kan niet meer gebruiken dan beschikbaar
            df.at[idx, 'Energy from Boiler (Wh)'] = energy_used
            
            # Update energieinhoud en temperatuur
            current_energy_content -= energy_used
            if max_energy_content > 0:  # Voorkom delen door nul
                current_temp = boiler.min_temp + (current_energy_content / max_energy_content) * (boiler.max_temp - boiler.min_temp)
            else:
                current_temp = boiler.min_temp
        else:
            df.at[idx, 'Energy from Boiler (Wh)'] = 0.0
        
        # Bereken warmteverlies
        heat_loss = boiler.calculate_heat_loss(current_temp, time_step_hours)
        df.at[idx, 'Heat Loss (Wh)'] = heat_loss
        
        # Update energieinhoud en temperatuur na warmteverlies
        current_energy_content = max(0, current_energy_content - heat_loss)
        if max_energy_content > 0:  # Voorkom delen door nul
            current_temp = boiler.min_temp + (current_energy_content / max_energy_content) * (boiler.max_temp - boiler.min_temp)
        else:
            current_temp = boiler.min_temp
        
        # Bepaal hoeveel energie naar de boiler kan
        net_energy = df.at[idx, 'Net Energy (Wh)']
        
        if net_energy > 0:  # Energieoverschot, laad de boiler op
            # Bereken hoeveel energie de boiler kan opnemen
            available_capacity = max_energy_content - current_energy_content
            max_power_this_step = boiler.heating_power * time_step_hours / 1000 * 1000  # Wh
            energy_to_boiler = min(net_energy, available_capacity, max_power_this_step)
            
            df.at[idx, 'Energy to Boiler (Wh)'] = energy_to_boiler
            
            # Update energieinhoud en temperatuur
            current_energy_content += energy_to_boiler * boiler.efficiency
            if max_energy_content > 0:  # Voorkom delen door nul
                current_temp = boiler.min_temp + (current_energy_content / max_energy_content) * (boiler.max_temp - boiler.min_temp)
            else:
                current_temp = boiler.min_temp
        else:
            df.at[idx, 'Energy to Boiler (Wh)'] = 0.0
    
    # Bereken statistieken
    total_energy_to_boiler = df['Energy to Boiler (Wh)'].sum()
    total_energy_from_boiler = df['Energy from Boiler (Wh)'].sum()
    total_heat_loss = df['Heat Loss (Wh)'].sum()
    total_hot_water_usage = df['Hot Water Usage (L)'].sum()
    
    # Bereken efficiëntie
    if total_energy_to_boiler > 0:
        boiler_efficiency = total_energy_from_boiler / (total_energy_to_boiler * boiler.efficiency) * 100
    else:
        boiler_efficiency = 0.0
    
    # Bereken financiële besparingen
    gas_cost = boiler.calculate_gas_cost(total_energy_from_boiler)
    electricity_cost = boiler.calculate_electricity_cost(total_energy_to_boiler)
    savings = gas_cost - electricity_cost
    
    # Bereken percentage van overschot dat is gebruikt
    total_surplus = data[data['Net Energy (Wh)'] > 0]['Net Energy (Wh)'].sum()
    surplus_utilization = (total_energy_to_boiler / total_surplus * 100) if total_surplus > 0 else 0.0
    
    # Stel resultaten samen
    results = {
        'simulation_data': df,
        'total_energy_to_boiler': total_energy_to_boiler,
        'total_energy_from_boiler': total_energy_from_boiler,
        'total_heat_loss': total_heat_loss,
        'total_hot_water_usage': total_hot_water_usage,
        'boiler_efficiency': boiler_efficiency,
        'gas_cost': gas_cost,
        'electricity_cost': electricity_cost,
        'savings': savings,
        'surplus_utilization': surplus_utilization,
        'boiler_model': boiler,
        'max_energy_content': max_energy_content
    }
    
    return results


def create_boiler_simulation_plots(simulation_results: Dict[str, Any]) -> Dict[str, go.Figure]:
    """Creëer visualisaties van de boilersimulatie.
    
    Args:
        simulation_results: Resultaten van de boilersimulatie
        
    Returns:
        Dict[str, go.Figure]: Dictionary met Plotly figuren
    """
    df = simulation_results['simulation_data']
    boiler = simulation_results['boiler_model']
    max_energy_content = simulation_results['max_energy_content']
    
    figures = {}
    
    # 1. Temperatuur en energieinhoud plot
    fig_temp = go.Figure()
    
    # Temperatuur
    fig_temp.add_trace(go.Scatter(
        x=df.index,
        y=df['Boiler Temperature (°C)'],
        mode='lines',
        name='Boilertemperatuur',
        line=dict(color='red')
    ))
    
    # Voeg horizontale lijnen toe voor min en max temperatuur
    fig_temp.add_shape(
        type="line",
        x0=df.index[0],
        y0=boiler.min_temp,
        x1=df.index[-1],
        y1=boiler.min_temp,
        line=dict(color="blue", width=1, dash="dash"),
    )
    
    fig_temp.add_shape(
        type="line",
        x0=df.index[0],
        y0=boiler.max_temp,
        x1=df.index[-1],
        y1=boiler.max_temp,
        line=dict(color="red", width=1, dash="dash"),
    )
    
    # Secundaire y-as voor energieinhoud
    fig_temp.add_trace(go.Scatter(
        x=df.index,
        y=df['Boiler Energy Content (Wh)'],
        mode='lines',
        name='Energieinhoud',
        line=dict(color='orange'),
        yaxis="y2"
    ))
    
    fig_temp.update_layout(
        title="Boilertemperatuur en Energieinhoud",
        xaxis_title="Datum/Tijd",
        yaxis_title="Temperatuur (°C)",
        yaxis2=dict(
            title="Energieinhoud (Wh)",
            overlaying="y",
            side="right",
            range=[0, max_energy_content * 1.1]
        ),
        hovermode="x unified"
    )
    
    figures['temperature_energy'] = fig_temp
    
    # 2. Energiestromen plot
    fig_energy = go.Figure()
    
    # Energie naar boiler
    fig_energy.add_trace(go.Scatter(
        x=df.index,
        y=df['Energy to Boiler (Wh)'],
        mode='lines',
        name='Energie naar Boiler',
        line=dict(color='green')
    ))
    
    # Energie uit boiler
    fig_energy.add_trace(go.Scatter(
        x=df.index,
        y=df['Energy from Boiler (Wh)'],
        mode='lines',
        name='Energie uit Boiler',
        line=dict(color='red')
    ))
    
    # Warmteverlies
    fig_energy.add_trace(go.Scatter(
        x=df.index,
        y=df['Heat Loss (Wh)'],
        mode='lines',
        name='Warmteverlies',
        line=dict(color='gray')
    ))
    
    fig_energy.update_layout(
        title="Energiestromen Boiler",
        xaxis_title="Datum/Tijd",
        yaxis_title="Energie (Wh)",
        hovermode="x unified"
    )
    
    figures['energy_flows'] = fig_energy
    
    # 3. Warmwatergebruik plot
    fig_usage = go.Figure()
    
    fig_usage.add_trace(go.Scatter(
        x=df.index,
        y=df['Hot Water Usage (L)'],
        mode='lines',
        name='Warmwatergebruik',
        line=dict(color='blue')
    ))
    
    fig_usage.update_layout(
        title="Warmwatergebruik",
        xaxis_title="Datum/Tijd",
        yaxis_title="Watergebruik (L)",
        hovermode="x unified"
    )
    
    figures['water_usage'] = fig_usage
    
    # 4. Vergelijking met energieoverschot
    fig_comparison = go.Figure()
    
    fig_comparison.add_trace(go.Scatter(
        x=df.index,
        y=df['Net Energy (Wh)'].clip(lower=0),  # Alleen positieve waarden (overschot)
        mode='lines',
        name='Energieoverschot',
        line=dict(color='lightgreen'),
        fill='tozeroy'
    ))
    
    fig_comparison.add_trace(go.Scatter(
        x=df.index,
        y=df['Energy to Boiler (Wh)'],
        mode='lines',
        name='Benut voor Boiler',
        line=dict(color='darkgreen')
    ))
    
    fig_comparison.update_layout(
        title="Benutting van Energieoverschot",
        xaxis_title="Datum/Tijd",
        yaxis_title="Energie (Wh)",
        hovermode="x unified"
    )
    
    figures['surplus_utilization'] = fig_comparison
    
    return figures


def calculate_payback_period(simulation_results: Dict[str, Any], 
                            boiler_cost: float, 
                            installation_cost: float = 0,
                            annual_maintenance: float = 0,
                            years: int = 15) -> Dict[str, Any]:
    """Bereken de terugverdientijd en financiële analyse voor een boilerinvestering.
    
    Args:
        simulation_results: Resultaten van de boilersimulatie
        boiler_cost: Aanschafkosten van de boiler in €
        installation_cost: Installatiekosten in €
        annual_maintenance: Jaarlijkse onderhoudskosten in €
        years: Aantal jaren voor de analyse
        
    Returns:
        Dict[str, Any]: Financiële analyse resultaten
    """
    # Haal simulatieperiode op
    df = simulation_results['simulation_data']
    simulation_days = (df.index[-1] - df.index[0]).total_seconds() / (24 * 3600)
    
    if simulation_days <= 0:
        raise ValueError("Simulatieperiode moet positief zijn")
    
    # Bereken jaarlijkse besparingen door extrapolatie
    daily_savings = simulation_results['savings'] / simulation_days
    annual_savings = daily_savings * 365
    
    # Totale investeringskosten
    total_investment = boiler_cost + installation_cost
    
    # Bereken cumulatieve kasstromen over de jaren
    cash_flows = []
    cumulative_cash_flow = -total_investment
    
    for year in range(1, years + 1):
        yearly_cash_flow = annual_savings - annual_maintenance
        cumulative_cash_flow += yearly_cash_flow
        cash_flows.append({
            'year': year,
            'yearly_savings': annual_savings,
            'yearly_maintenance': annual_maintenance,
            'yearly_cash_flow': yearly_cash_flow,
            'cumulative_cash_flow': cumulative_cash_flow
        })
    
    # Bereken terugverdientijd in jaren
    if annual_savings <= annual_maintenance:
        payback_period = float('inf')  # Nooit terugverdiend
    else:
        payback_period = total_investment / (annual_savings - annual_maintenance)
    
    # Bereken netto contante waarde (NCW) met een discontovoet van 3%
    discount_rate = 0.03
    npv = -total_investment
    
    for year in range(1, years + 1):
        yearly_cash_flow = annual_savings - annual_maintenance
        npv += yearly_cash_flow / ((1 + discount_rate) ** year)
    
    # Bereken return on investment (ROI) na de geanalyseerde periode
    if total_investment > 0:
        roi = (cumulative_cash_flow + total_investment) / total_investment * 100
    else:
        roi = 0.0
    
    # Stel resultaten samen
    results = {
        'annual_savings': annual_savings,
        'total_investment': total_investment,
        'payback_period': payback_period,
        'npv': npv,
        'roi': roi,
        'cash_flows': cash_flows,
        'simulation_days': simulation_days
    }
    
    return results


def create_financial_analysis_plot(financial_results: Dict[str, Any]) -> go.Figure:
    """Creëer een visualisatie van de financiële analyse.
    
    Args:
        financial_results: Resultaten van de financiële analyse
        
    Returns:
        go.Figure: Plotly figuur met de financiële analyse
    """
    cash_flows = financial_results['cash_flows']
    
    # Maak een DataFrame van de kasstromen
    df = pd.DataFrame(cash_flows)
    
    # Creëer de figuur
    fig = go.Figure()
    
    # Voeg cumulatieve kasstroom toe
    fig.add_trace(go.Scatter(
        x=df['year'],
        y=df['cumulative_cash_flow'],
        mode='lines+markers',
        name='Cumulatieve Kasstroom',
        line=dict(color='blue')
    ))
    
    # Voeg horizontale lijn toe bij y=0 (terugverdienmoment)
    fig.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=df['year'].max(),
        y1=0,
        line=dict(color="red", width=1, dash="dash"),
    )
    
    # Voeg verticale lijn toe bij terugverdientijd
    if financial_results['payback_period'] <= df['year'].max():
        fig.add_shape(
            type="line",
            x0=financial_results['payback_period'],
            y0=df['cumulative_cash_flow'].min(),
            x1=financial_results['payback_period'],
            y1=df['cumulative_cash_flow'].max(),
            line=dict(color="green", width=1, dash="dash"),
        )
        
        fig.add_annotation(
            x=financial_results['payback_period'],
            y=0,
            text=f"Terugverdientijd: {financial_results['payback_period']:.1f} jaar",
            showarrow=True,
            arrowhead=1
        )
    
    # Update layout
    fig.update_layout(
        title="Financiële Analyse Warmwaterboiler",
        xaxis_title="Jaar",
        yaxis_title="Cumulatieve Kasstroom (€)",
        hovermode="x unified"
    )
    
    return fig


def optimize_boiler_parameters(data: pd.DataFrame, 
                              volume_range: List[float] = [100, 150, 200, 300],
                              usage_profiles: Dict[str, Dict[int, float]] = None) -> Dict[str, Any]:
    """Optimaliseer boilerparameters voor maximale besparingen.
    
    Args:
        data: DataFrame met energiedata
        volume_range: Lijst met te evalueren boilervolumes in liters
        usage_profiles: Dictionary met verschillende gebruiksprofielen om te evalueren
        
    Returns:
        Dict[str, Any]: Resultaten van de optimalisatie
    """
    if usage_profiles is None:
        # Standaard gebruiksprofielen
        usage_profiles = {
            "Laag gebruik": {7: 30, 19: 30},
            "Gemiddeld gebruik": {7: 50, 8: 20, 19: 40, 22: 20},
            "Hoog gebruik": {7: 60, 8: 30, 13: 20, 19: 50, 22: 40}
        }
    
    results = []
    
    for volume in volume_range:
        for profile_name, profile in usage_profiles.items():
            # Maak een boilermodel met het huidige volume
            boiler = BoilerModel(volume=volume)
            
            # Simuleer met het huidige gebruiksprofiel
            simulation = simulate_boiler_storage(data, boiler, usage_profile=profile)
            
            # Bereken financiële resultaten (aanname: €500 basiskosten + €2 per liter)
            boiler_cost = 500 + 2 * volume
            financial = calculate_payback_period(simulation, boiler_cost=boiler_cost)
            
            # Sla resultaten op
            results.append({
                'volume': volume,
                'profile_name': profile_name,
                'annual_savings': financial['annual_savings'],
                'payback_period': financial['payback_period'],
                'npv': financial['npv'],
                'roi': financial['roi'],
                'boiler_cost': boiler_cost,
                'total_energy_to_boiler': simulation['total_energy_to_boiler'],
                'total_energy_from_boiler': simulation['total_energy_from_boiler'],
                'boiler_efficiency': simulation['boiler_efficiency'],
                'surplus_utilization': simulation['surplus_utilization']
            })
    
    # Converteer naar DataFrame voor eenvoudigere analyse
    results_df = pd.DataFrame(results)
    
    # Vind de optimale configuratie (hoogste NPV)
    optimal_config = results_df.loc[results_df['npv'].idxmax()]
    
    # Stel resultaten samen
    optimization_results = {
        'all_results': results_df,
        'optimal_config': optimal_config,
        'volume_range': volume_range,
        'usage_profiles': usage_profiles
    }
    
    return optimization_results


def create_optimization_plots(optimization_results: Dict[str, Any]) -> Dict[str, go.Figure]:
    """Creëer visualisaties van de optimalisatieresultaten.
    
    Args:
        optimization_results: Resultaten van de boileroptimalisatie
        
    Returns:
        Dict[str, go.Figure]: Dictionary met Plotly figuren
    """
    df = optimization_results['all_results']
    
    figures = {}
    
    # 1. Terugverdientijd per volume en gebruiksprofiel
    fig_payback = px.bar(
        df,
        x='volume',
        y='payback_period',
        color='profile_name',
        barmode='group',
        title="Terugverdientijd per Boilervolume en Gebruiksprofiel",
        labels={
            'volume': 'Boilervolume (L)',
            'payback_period': 'Terugverdientijd (jaren)',
            'profile_name': 'Gebruiksprofiel'
        }
    )
    
    figures['payback_comparison'] = fig_payback
    
    # 2. Jaarlijkse besparingen per volume en gebruiksprofiel
    fig_savings = px.bar(
        df,
        x='volume',
        y='annual_savings',
        color='profile_name',
        barmode='group',
        title="Jaarlijkse Besparingen per Boilervolume en Gebruiksprofiel",
        labels={
            'volume': 'Boilervolume (L)',
            'annual_savings': 'Jaarlijkse Besparingen (€)',
            'profile_name': 'Gebruiksprofiel'
        }
    )
    
    figures['savings_comparison'] = fig_savings
    
    # 3. Efficiëntie en surplus benutting per volume
    fig_efficiency = go.Figure()
    
    for profile in df['profile_name'].unique():
        profile_df = df[df['profile_name'] == profile]
        
        fig_efficiency.add_trace(go.Scatter(
            x=profile_df['volume'],
            y=profile_df['boiler_efficiency'],
            mode='lines+markers',
            name=f'Efficiëntie - {profile}'
        ))
        
        fig_efficiency.add_trace(go.Scatter(
            x=profile_df['volume'],
            y=profile_df['surplus_utilization'],
            mode='lines+markers',
            name=f'Surplus Benutting - {profile}',
            line=dict(dash='dash')
        ))
    
    fig_efficiency.update_layout(
        title="Boilerefficiëntie en Surplus Benutting per Volume",
        xaxis_title="Boilervolume (L)",
        yaxis_title="Percentage (%)",
        hovermode="x unified"
    )
    
    figures['efficiency_comparison'] = fig_efficiency
    
    return figures