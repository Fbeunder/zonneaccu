"""Module voor berekeningen gerelateerd aan accu/batterij energieopslag.

Deze module bevat functies voor:
- Het modelleren van een accu als energieopslagmedium
- Het berekenen van energieopslagcapaciteit en verliezen
- Het simuleren van energieopslag in een accu
- Het analyseren van besparingen door gebruik van een accu
- Het berekenen van degradatie en levensduur
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, Optional, Union, List, Tuple
from datetime import datetime, timedelta


class BatteryModel:
    """Model van een accu voor energieopslag.
    
    Deze klasse bevat de eigenschappen en methoden voor het modelleren van een
    accu als energieopslagmedium, inclusief berekeningen voor
    energieopslagcapaciteit, laad/ontlaadverliezen, degradatie en efficiëntie.
    """
    
    def __init__(self, 
                 capacity: float = 10.0,  # kWh
                 max_charge_rate: float = 3.7,  # kW
                 max_discharge_rate: float = 3.7,  # kW
                 charge_efficiency: float = 0.95,  # 95%
                 discharge_efficiency: float = 0.95,  # 95%
                 self_discharge_rate: float = 0.002,  # 0.2% per dag
                 min_soc: float = 0.1,  # 10%
                 max_soc: float = 0.9,  # 90%
                 cycle_degradation_rate: float = 0.0005,  # 0.05% per cyclus
                 calendar_degradation_rate: float = 0.02,  # 2% per jaar
                 max_cycles: int = 5000,  # maximaal aantal cycli
                 electricity_buy_price: float = 0.30,  # €/kWh
                 electricity_sell_price: float = 0.15,  # €/kWh
                ):
        """Initialiseer een BatteryModel met de gegeven parameters.
        
        Args:
            capacity: Capaciteit van de accu in kWh
            max_charge_rate: Maximale laadsnelheid in kW
            max_discharge_rate: Maximale ontlaadsnelheid in kW
            charge_efficiency: Efficiëntie van het laadproces (0-1)
            discharge_efficiency: Efficiëntie van het ontlaadproces (0-1)
            self_discharge_rate: Zelfontladingspercentage per dag (0-1)
            min_soc: Minimale State of Charge (0-1)
            max_soc: Maximale State of Charge (0-1)
            cycle_degradation_rate: Degradatie per volledige cyclus (0-1)
            calendar_degradation_rate: Degradatie per jaar (0-1)
            max_cycles: Maximaal aantal volledige cycli
            electricity_buy_price: Inkoopprijs elektriciteit in €/kWh
            electricity_sell_price: Verkoopprijs elektriciteit in €/kWh
        """
        self.capacity = capacity  # kWh
        self.max_charge_rate = max_charge_rate  # kW
        self.max_discharge_rate = max_discharge_rate  # kW
        self.charge_efficiency = charge_efficiency  # fractie (0-1)
        self.discharge_efficiency = discharge_efficiency  # fractie (0-1)
        self.self_discharge_rate = self_discharge_rate  # fractie per dag (0-1)
        self.min_soc = min_soc  # fractie (0-1)
        self.max_soc = max_soc  # fractie (0-1)
        self.cycle_degradation_rate = cycle_degradation_rate  # fractie per cyclus (0-1)
        self.calendar_degradation_rate = calendar_degradation_rate  # fractie per jaar (0-1)
        self.max_cycles = max_cycles  # aantal
        self.electricity_buy_price = electricity_buy_price  # €/kWh
        self.electricity_sell_price = electricity_sell_price  # €/kWh
        
        # Bereken de bruikbare capaciteit
        self.usable_capacity = self.capacity * (self.max_soc - self.min_soc)  # kWh
    
    def calculate_storage_capacity(self) -> float:
        """Bereken de bruikbare energieopslagcapaciteit van de accu in Wh.
        
        Returns:
            float: Bruikbare opslagcapaciteit in Wh
        """
        return self.usable_capacity * 1000  # Wh
    
    def calculate_self_discharge(self, current_energy: float, days: float) -> float:
        """Bereken het energieverlies door zelfontlading over een bepaalde periode.
        
        Args:
            current_energy: Huidige energieinhoud in Wh
            days: Aantal dagen
            
        Returns:
            float: Energieverlies door zelfontlading in Wh
        """
        # Bereken dagelijkse zelfontlading
        daily_loss_rate = self.self_discharge_rate  # fractie per dag
        
        # Bereken totale zelfontlading over de periode
        total_loss_fraction = 1 - (1 - daily_loss_rate) ** days
        energy_loss = current_energy * total_loss_fraction  # Wh
        
        return energy_loss
    
    def calculate_cycle_degradation(self, energy_throughput: float) -> float:
        """Bereken de capaciteitsdegradatie op basis van energiedoorvoer.
        
        Args:
            energy_throughput: Totale energiedoorvoer in Wh
            
        Returns:
            float: Degradatiefractie (0-1)
        """
        # Bereken het equivalent aantal volledige cycli
        full_cycles = energy_throughput / (self.capacity * 1000 * 2)  # Delen door 2 omdat een cyclus = laden + ontladen
        
        # Bereken degradatie
        degradation = full_cycles * self.cycle_degradation_rate
        
        return degradation
    
    def calculate_calendar_degradation(self, years: float) -> float:
        """Bereken de capaciteitsdegradatie op basis van tijd.
        
        Args:
            years: Aantal jaren
            
        Returns:
            float: Degradatiefractie (0-1)
        """
        # Bereken degradatie
        degradation = years * self.calendar_degradation_rate
        
        return degradation
    
    def calculate_remaining_cycles(self, current_degradation: float) -> float:
        """Bereken het resterende aantal cycli op basis van huidige degradatie.
        
        Args:
            current_degradation: Huidige degradatiefractie (0-1)
            
        Returns:
            float: Resterend aantal volledige cycli
        """
        if self.cycle_degradation_rate <= 0:
            return float('inf')  # Voorkom delen door nul
        
        # Bereken maximale degradatie door cycli (meestal 0.2 = 20%)
        max_cycle_degradation = 0.2
        
        # Bereken resterende degradatie
        remaining_degradation = max_cycle_degradation - current_degradation
        
        # Bereken resterende cycli
        remaining_cycles = remaining_degradation / self.cycle_degradation_rate
        
        return max(0, remaining_cycles)
    
    def calculate_charge_time(self, energy_to_charge: float) -> float:
        """Bereken de tijd nodig om een bepaalde hoeveelheid energie op te laden.
        
        Args:
            energy_to_charge: Op te laden energie in Wh
            
        Returns:
            float: Laadtijd in uren
        """
        if self.max_charge_rate <= 0:
            return float('inf')  # Voorkom delen door nul
        
        # Bereken laadtijd, rekening houdend met efficiëntie
        charge_time = energy_to_charge / (self.max_charge_rate * 1000 * self.charge_efficiency)
        
        return charge_time
    
    def calculate_discharge_time(self, energy_to_discharge: float) -> float:
        """Bereken de tijd nodig om een bepaalde hoeveelheid energie te ontladen.
        
        Args:
            energy_to_discharge: Te ontladen energie in Wh
            
        Returns:
            float: Ontlaadtijd in uren
        """
        if self.max_discharge_rate <= 0:
            return float('inf')  # Voorkom delen door nul
        
        # Bereken ontlaadtijd, rekening houdend met efficiëntie
        discharge_time = energy_to_discharge / (self.max_discharge_rate * 1000)
        
        return discharge_time
    
    def calculate_savings(self, energy_stored: float, energy_used: float) -> float:
        """Bereken de besparingen door energie op te slaan en later te gebruiken.
        
        Args:
            energy_stored: Opgeslagen energie in Wh
            energy_used: Gebruikte energie in Wh
            
        Returns:
            float: Besparingen in €
        """
        # Bereken kosten als de energie direct zou worden verkocht
        direct_sell_value = (energy_stored / 1000) * self.electricity_sell_price  # €
        
        # Bereken waarde van de gebruikte energie (vermeden inkoop)
        used_energy_value = (energy_used / 1000) * self.electricity_buy_price  # €
        
        # Bereken besparingen
        savings = used_energy_value - direct_sell_value
        
        return savings


def simulate_battery_storage(data: pd.DataFrame, 
                           battery: BatteryModel,
                           usage_profile: Dict[int, float] = None,
                           initial_soc: float = None,
                           control_strategy: str = "maximize_self_consumption") -> Dict[str, Any]:
    """Simuleer energieopslag in een accu op basis van energiedata.
    
    Args:
        data: DataFrame met energiedata met een DatetimeIndex
        battery: BatteryModel object met accu-eigenschappen
        usage_profile: Dictionary met uren (0-23) als keys en extra verbruik in Wh als values
        initial_soc: Initiële State of Charge (0-1) (standaard: battery.min_soc)
        control_strategy: Strategie voor het laden/ontladen van de accu
                         ("maximize_self_consumption", "time_of_use", "peak_shaving")
        
    Returns:
        Dict[str, Any]: Resultaten van de simulatie
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Data moet een DatetimeIndex hebben voor accusimulatie")
    
    # Controleer of de benodigde kolommen aanwezig zijn
    required_columns = ['Net Energy (Wh)', 'Energy Produced (Wh)', 'Energy Consumed (Wh)']
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Kolom '{col}' ontbreekt in de data")
    
    # Standaard gebruiksprofiel als er geen is opgegeven
    if usage_profile is None:
        # Standaard: geen extra verbruik
        usage_profile = {}
    
    # Initiële State of Charge
    if initial_soc is None:
        initial_soc = battery.min_soc
    
    # Maak een kopie van de data voor simulatie
    df = data.copy()
    
    # Voeg kolommen toe voor accusimulatie
    df['Battery SoC'] = 0.0
    df['Battery Energy Content (Wh)'] = 0.0
    df['Energy to Battery (Wh)'] = 0.0
    df['Energy from Battery (Wh)'] = 0.0
    df['Self Discharge (Wh)'] = 0.0
    df['Extra Consumption (Wh)'] = 0.0
    df['Grid Import (Wh)'] = 0.0
    df['Grid Export (Wh)'] = 0.0
    
    # Bereken tijdstap in uren
    if len(df) > 1:
        time_step_hours = (df.index[1] - df.index[0]).total_seconds() / 3600
    else:
        time_step_hours = 1.0  # Standaard 1 uur als er maar één rij is
    
    # Bereken tijdstap in dagen voor zelfontlading
    time_step_days = time_step_hours / 24
    
    # Bereken maximale energieinhoud van de accu
    max_energy_content = battery.calculate_storage_capacity()  # Wh
    
    # Initialiseer simulatievariabelen
    current_soc = initial_soc
    current_energy_content = current_soc * max_energy_content  # Wh
    total_energy_throughput = 0.0  # Wh, voor degradatieberekening
    
    # Simuleer voor elke tijdstap
    for i, idx in enumerate(df.index):
        # Sla huidige status op
        df.at[idx, 'Battery SoC'] = current_soc
        df.at[idx, 'Battery Energy Content (Wh)'] = current_energy_content
        
        # Bepaal extra verbruik op basis van het gebruiksprofiel
        hour = idx.hour
        extra_consumption = usage_profile.get(hour, 0.0)  # Wh
        df.at[idx, 'Extra Consumption (Wh)'] = extra_consumption
        
        # Bereken zelfontlading
        self_discharge = battery.calculate_self_discharge(current_energy_content, time_step_days)
        df.at[idx, 'Self Discharge (Wh)'] = self_discharge
        
        # Update energieinhoud en SoC na zelfontlading
        current_energy_content = max(0, current_energy_content - self_discharge)
        if max_energy_content > 0:  # Voorkom delen door nul
            current_soc = current_energy_content / max_energy_content
        else:
            current_soc = battery.min_soc
        
        # Bepaal netto energie en extra verbruik
        net_energy = df.at[idx, 'Net Energy (Wh)'] - extra_consumption
        
        # Implementeer de gekozen besturingsstrategie
        if control_strategy == "maximize_self_consumption":
            # Maximaliseer eigen verbruik: laad bij overschot, ontlaad bij tekort
            if net_energy > 0:  # Energieoverschot, laad de accu op
                # Bereken hoeveel energie de accu kan opnemen
                available_capacity = max_energy_content * (battery.max_soc - current_soc)
                max_charge_this_step = battery.max_charge_rate * time_step_hours * 1000  # Wh
                energy_to_battery = min(net_energy, available_capacity, max_charge_this_step)
                
                # Bereken werkelijk opgeslagen energie (rekening houdend met laadefficiëntie)
                stored_energy = energy_to_battery * battery.charge_efficiency
                
                df.at[idx, 'Energy to Battery (Wh)'] = energy_to_battery
                df.at[idx, 'Grid Export (Wh)'] = net_energy - energy_to_battery
                
                # Update energieinhoud, SoC en energiedoorvoer
                current_energy_content += stored_energy
                current_soc = current_energy_content / max_energy_content
                total_energy_throughput += energy_to_battery
                
            elif net_energy < 0:  # Energietekort, ontlaad de accu
                # Bereken hoeveel energie de accu kan leveren
                available_energy = current_energy_content - (battery.min_soc * max_energy_content)
                max_discharge_this_step = battery.max_discharge_rate * time_step_hours * 1000  # Wh
                energy_needed = -net_energy  # Maak positief
                
                # Bereken werkelijk te ontladen energie (rekening houdend met ontlaadefficiëntie)
                energy_from_battery = min(available_energy, max_discharge_this_step, energy_needed / battery.discharge_efficiency)
                
                # Bereken werkelijk geleverde energie
                delivered_energy = energy_from_battery * battery.discharge_efficiency
                
                df.at[idx, 'Energy from Battery (Wh)'] = delivered_energy
                df.at[idx, 'Grid Import (Wh)'] = max(0, energy_needed - delivered_energy)
                
                # Update energieinhoud, SoC en energiedoorvoer
                current_energy_content -= energy_from_battery
                current_soc = current_energy_content / max_energy_content
                total_energy_throughput += energy_from_battery
                
            else:  # Geen overschot of tekort
                df.at[idx, 'Energy to Battery (Wh)'] = 0.0
                df.at[idx, 'Energy from Battery (Wh)'] = 0.0
                df.at[idx, 'Grid Import (Wh)'] = 0.0
                df.at[idx, 'Grid Export (Wh)'] = 0.0
                
        elif control_strategy == "time_of_use":
            # Time-of-use optimalisatie: laad tijdens daluren, ontlaad tijdens piekuren
            # Definieer piek- en daluren (voorbeeld: piek 17-21 uur, dal 0-6 uur)
            peak_hours = range(17, 22)  # 17:00 - 21:59
            off_peak_hours = range(0, 7)  # 00:00 - 06:59
            
            if hour in off_peak_hours:  # Daluren, prioriteit aan laden
                # Bereken hoeveel energie de accu kan opnemen
                available_capacity = max_energy_content * (battery.max_soc - current_soc)
                max_charge_this_step = battery.max_charge_rate * time_step_hours * 1000  # Wh
                
                # Tijdens daluren, importeer energie van het net om de accu te laden
                energy_to_import = min(available_capacity, max_charge_this_step)
                energy_to_battery = energy_to_import
                
                if net_energy > 0:  # Als er ook overschot is, gebruik dat eerst
                    energy_from_grid = max(0, energy_to_battery - net_energy)
                    energy_to_export = max(0, net_energy - energy_to_battery)
                else:
                    energy_from_grid = energy_to_battery - net_energy  # Importeer extra
                    energy_to_export = 0
                
                # Bereken werkelijk opgeslagen energie
                stored_energy = energy_to_battery * battery.charge_efficiency
                
                df.at[idx, 'Energy to Battery (Wh)'] = energy_to_battery
                df.at[idx, 'Grid Import (Wh)'] = energy_from_grid
                df.at[idx, 'Grid Export (Wh)'] = energy_to_export
                
                # Update energieinhoud, SoC en energiedoorvoer
                current_energy_content += stored_energy
                current_soc = current_energy_content / max_energy_content
                total_energy_throughput += energy_to_battery
                
            elif hour in peak_hours:  # Piekuren, prioriteit aan ontladen
                # Bereken hoeveel energie de accu kan leveren
                available_energy = current_energy_content - (battery.min_soc * max_energy_content)
                max_discharge_this_step = battery.max_discharge_rate * time_step_hours * 1000  # Wh
                
                # Tijdens piekuren, ontlaad de accu maximaal
                energy_from_battery = min(available_energy, max_discharge_this_step)
                
                # Bereken werkelijk geleverde energie
                delivered_energy = energy_from_battery * battery.discharge_efficiency
                
                if net_energy < 0:  # Energietekort
                    energy_needed = -net_energy  # Maak positief
                    energy_from_grid = max(0, energy_needed - delivered_energy)
                    energy_to_export = 0
                else:  # Energieoverschot
                    energy_from_grid = 0
                    energy_to_export = net_energy + delivered_energy
                
                df.at[idx, 'Energy from Battery (Wh)'] = delivered_energy
                df.at[idx, 'Grid Import (Wh)'] = energy_from_grid
                df.at[idx, 'Grid Export (Wh)'] = energy_to_export
                
                # Update energieinhoud, SoC en energiedoorvoer
                current_energy_content -= energy_from_battery
                current_soc = current_energy_content / max_energy_content
                total_energy_throughput += energy_from_battery
                
            else:  # Normale uren, gedrag zoals bij maximize_self_consumption
                if net_energy > 0:  # Energieoverschot, laad de accu op
                    # Bereken hoeveel energie de accu kan opnemen
                    available_capacity = max_energy_content * (battery.max_soc - current_soc)
                    max_charge_this_step = battery.max_charge_rate * time_step_hours * 1000  # Wh
                    energy_to_battery = min(net_energy, available_capacity, max_charge_this_step)
                    
                    # Bereken werkelijk opgeslagen energie
                    stored_energy = energy_to_battery * battery.charge_efficiency
                    
                    df.at[idx, 'Energy to Battery (Wh)'] = energy_to_battery
                    df.at[idx, 'Grid Export (Wh)'] = net_energy - energy_to_battery
                    
                    # Update energieinhoud, SoC en energiedoorvoer
                    current_energy_content += stored_energy
                    current_soc = current_energy_content / max_energy_content
                    total_energy_throughput += energy_to_battery
                    
                elif net_energy < 0:  # Energietekort, ontlaad de accu
                    # Bereken hoeveel energie de accu kan leveren
                    available_energy = current_energy_content - (battery.min_soc * max_energy_content)
                    max_discharge_this_step = battery.max_discharge_rate * time_step_hours * 1000  # Wh
                    energy_needed = -net_energy  # Maak positief
                    
                    # Bereken werkelijk te ontladen energie
                    energy_from_battery = min(available_energy, max_discharge_this_step, energy_needed / battery.discharge_efficiency)
                    
                    # Bereken werkelijk geleverde energie
                    delivered_energy = energy_from_battery * battery.discharge_efficiency
                    
                    df.at[idx, 'Energy from Battery (Wh)'] = delivered_energy
                    df.at[idx, 'Grid Import (Wh)'] = max(0, energy_needed - delivered_energy)
                    
                    # Update energieinhoud, SoC en energiedoorvoer
                    current_energy_content -= energy_from_battery
                    current_soc = current_energy_content / max_energy_content
                    total_energy_throughput += energy_from_battery
                    
                else:  # Geen overschot of tekort
                    df.at[idx, 'Energy to Battery (Wh)'] = 0.0
                    df.at[idx, 'Energy from Battery (Wh)'] = 0.0
                    df.at[idx, 'Grid Import (Wh)'] = 0.0
                    df.at[idx, 'Grid Export (Wh)'] = 0.0
                    
        elif control_strategy == "peak_shaving":
            # Peak shaving: verminder pieken in import en export
            # Bereken gemiddelde en standaarddeviatie van historische netto energie
            if i > 24:  # Wacht tot we genoeg historische data hebben
                historical_window = 24  # Gebruik 24 tijdstappen voor het bepalen van de drempel
                historical_net = df['Net Energy (Wh)'].iloc[max(0, i-historical_window):i].values
                avg_net = np.mean(historical_net)
                std_net = np.std(historical_net)
                
                # Definieer drempels voor piekdetectie
                import_threshold = avg_net - std_net  # Negatieve waarde voor import
                export_threshold = avg_net + std_net  # Positieve waarde voor export
                
                if net_energy < import_threshold:  # Importpiek, ontlaad de accu
                    # Bereken hoeveel energie de accu kan leveren
                    available_energy = current_energy_content - (battery.min_soc * max_energy_content)
                    max_discharge_this_step = battery.max_discharge_rate * time_step_hours * 1000  # Wh
                    energy_needed = -net_energy + import_threshold  # Hoeveel we willen verminderen
                    
                    # Bereken werkelijk te ontladen energie
                    energy_from_battery = min(available_energy, max_discharge_this_step, energy_needed / battery.discharge_efficiency)
                    
                    # Bereken werkelijk geleverde energie
                    delivered_energy = energy_from_battery * battery.discharge_efficiency
                    
                    df.at[idx, 'Energy from Battery (Wh)'] = delivered_energy
                    df.at[idx, 'Grid Import (Wh)'] = -net_energy - delivered_energy
                    
                    # Update energieinhoud, SoC en energiedoorvoer
                    current_energy_content -= energy_from_battery
                    current_soc = current_energy_content / max_energy_content
                    total_energy_throughput += energy_from_battery
                    
                elif net_energy > export_threshold:  # Exportpiek, laad de accu op
                    # Bereken hoeveel energie de accu kan opnemen
                    available_capacity = max_energy_content * (battery.max_soc - current_soc)
                    max_charge_this_step = battery.max_charge_rate * time_step_hours * 1000  # Wh
                    energy_to_store = net_energy - export_threshold  # Hoeveel we willen opslaan
                    
                    # Bereken werkelijk op te slaan energie
                    energy_to_battery = min(energy_to_store, available_capacity, max_charge_this_step)
                    
                    # Bereken werkelijk opgeslagen energie
                    stored_energy = energy_to_battery * battery.charge_efficiency
                    
                    df.at[idx, 'Energy to Battery (Wh)'] = energy_to_battery
                    df.at[idx, 'Grid Export (Wh)'] = net_energy - energy_to_battery
                    
                    # Update energieinhoud, SoC en energiedoorvoer
                    current_energy_content += stored_energy
                    current_soc = current_energy_content / max_energy_content
                    total_energy_throughput += energy_to_battery
                    
                else:  # Normale situatie, geen piek
                    # Standaardgedrag: laad bij overschot, ontlaad bij tekort, maar met minder prioriteit
                    if net_energy > 0 and current_soc < battery.max_soc * 0.9:  # Laad alleen als SoC < 90% van max
                        # Bereken hoeveel energie de accu kan opnemen
                        available_capacity = max_energy_content * (battery.max_soc - current_soc)
                        max_charge_this_step = battery.max_charge_rate * 0.5 * time_step_hours * 1000  # 50% van max laadsnelheid
                        energy_to_battery = min(net_energy, available_capacity, max_charge_this_step)
                        
                        # Bereken werkelijk opgeslagen energie
                        stored_energy = energy_to_battery * battery.charge_efficiency
                        
                        df.at[idx, 'Energy to Battery (Wh)'] = energy_to_battery
                        df.at[idx, 'Grid Export (Wh)'] = net_energy - energy_to_battery
                        
                        # Update energieinhoud, SoC en energiedoorvoer
                        current_energy_content += stored_energy
                        current_soc = current_energy_content / max_energy_content
                        total_energy_throughput += energy_to_battery
                        
                    elif net_energy < 0 and current_soc > battery.min_soc * 1.1:  # Ontlaad alleen als SoC > 110% van min
                        # Bereken hoeveel energie de accu kan leveren
                        available_energy = current_energy_content - (battery.min_soc * max_energy_content)
                        max_discharge_this_step = battery.max_discharge_rate * 0.5 * time_step_hours * 1000  # 50% van max ontlaadsnelheid
                        energy_needed = -net_energy  # Maak positief
                        
                        # Bereken werkelijk te ontladen energie
                        energy_from_battery = min(available_energy, max_discharge_this_step, energy_needed / battery.discharge_efficiency)
                        
                        # Bereken werkelijk geleverde energie
                        delivered_energy = energy_from_battery * battery.discharge_efficiency
                        
                        df.at[idx, 'Energy from Battery (Wh)'] = delivered_energy
                        df.at[idx, 'Grid Import (Wh)'] = max(0, energy_needed - delivered_energy)
                        
                        # Update energieinhoud, SoC en energiedoorvoer
                        current_energy_content -= energy_from_battery
                        current_soc = current_energy_content / max_energy_content
                        total_energy_throughput += energy_from_battery
                        
                    else:  # Geen actie
                        df.at[idx, 'Energy to Battery (Wh)'] = 0.0
                        df.at[idx, 'Energy from Battery (Wh)'] = 0.0
                        if net_energy > 0:
                            df.at[idx, 'Grid Export (Wh)'] = net_energy
                            df.at[idx, 'Grid Import (Wh)'] = 0.0
                        else:
                            df.at[idx, 'Grid Export (Wh)'] = 0.0
                            df.at[idx, 'Grid Import (Wh)'] = -net_energy
            else:
                # Niet genoeg historische data, gebruik standaardgedrag
                if net_energy > 0:  # Energieoverschot, laad de accu op
                    # Bereken hoeveel energie de accu kan opnemen
                    available_capacity = max_energy_content * (battery.max_soc - current_soc)
                    max_charge_this_step = battery.max_charge_rate * time_step_hours * 1000  # Wh
                    energy_to_battery = min(net_energy, available_capacity, max_charge_this_step)
                    
                    # Bereken werkelijk opgeslagen energie
                    stored_energy = energy_to_battery * battery.charge_efficiency
                    
                    df.at[idx, 'Energy to Battery (Wh)'] = energy_to_battery
                    df.at[idx, 'Grid Export (Wh)'] = net_energy - energy_to_battery
                    
                    # Update energieinhoud, SoC en energiedoorvoer
                    current_energy_content += stored_energy
                    current_soc = current_energy_content / max_energy_content
                    total_energy_throughput += energy_to_battery
                    
                elif net_energy < 0:  # Energietekort, ontlaad de accu
                    # Bereken hoeveel energie de accu kan leveren
                    available_energy = current_energy_content - (battery.min_soc * max_energy_content)
                    max_discharge_this_step = battery.max_discharge_rate * time_step_hours * 1000  # Wh
                    energy_needed = -net_energy  # Maak positief
                    
                    # Bereken werkelijk te ontladen energie
                    energy_from_battery = min(available_energy, max_discharge_this_step, energy_needed / battery.discharge_efficiency)
                    
                    # Bereken werkelijk geleverde energie
                    delivered_energy = energy_from_battery * battery.discharge_efficiency
                    
                    df.at[idx, 'Energy from Battery (Wh)'] = delivered_energy
                    df.at[idx, 'Grid Import (Wh)'] = max(0, energy_needed - delivered_energy)
                    
                    # Update energieinhoud, SoC en energiedoorvoer
                    current_energy_content -= energy_from_battery
                    current_soc = current_energy_content / max_energy_content
                    total_energy_throughput += energy_from_battery
                    
                else:  # Geen overschot of tekort
                    df.at[idx, 'Energy to Battery (Wh)'] = 0.0
                    df.at[idx, 'Energy from Battery (Wh)'] = 0.0
                    df.at[idx, 'Grid Import (Wh)'] = 0.0
                    df.at[idx, 'Grid Export (Wh)'] = 0.0
    
    # Bereken statistieken
    total_energy_to_battery = df['Energy to Battery (Wh)'].sum()
    total_energy_from_battery = df['Energy from Battery (Wh)'].sum()
    total_self_discharge = df['Self Discharge (Wh)'].sum()
    total_extra_consumption = df['Extra Consumption (Wh)'].sum()
    total_grid_import = df['Grid Import (Wh)'].sum()
    total_grid_export = df['Grid Export (Wh)'].sum()
    
    # Bereken round-trip efficiëntie
    if total_energy_to_battery > 0:
        round_trip_efficiency = total_energy_from_battery / total_energy_to_battery * 100
    else:
        round_trip_efficiency = 0.0
    
    # Bereken financiële besparingen
    # 1. Kosten zonder accu
    original_import_cost = data[data['Net Energy (Wh)'] < 0]['Net Energy (Wh)'].sum() / -1000 * battery.electricity_buy_price
    original_export_revenue = data[data['Net Energy (Wh)'] > 0]['Net Energy (Wh)'].sum() / 1000 * battery.electricity_sell_price
    original_net_cost = original_import_cost - original_export_revenue
    
    # 2. Kosten met accu
    new_import_cost = total_grid_import / 1000 * battery.electricity_buy_price
    new_export_revenue = total_grid_export / 1000 * battery.electricity_sell_price
    new_net_cost = new_import_cost - new_export_revenue
    
    # 3. Besparingen
    savings = original_net_cost - new_net_cost
    
    # Bereken zelfconsumptie percentage
    total_production = data['Energy Produced (Wh)'].sum()
    
    if total_production > 0:
        original_self_consumption = (total_production - data[data['Net Energy (Wh)'] > 0]['Net Energy (Wh)'].sum()) / total_production * 100
        new_self_consumption = (total_production - total_grid_export) / total_production * 100
        self_consumption_increase = new_self_consumption - original_self_consumption
    else:
        original_self_consumption = 0.0
        new_self_consumption = 0.0
        self_consumption_increase = 0.0
    
    # Bereken degradatie
    simulation_days = (df.index[-1] - df.index[0]).total_seconds() / (24 * 3600)
    simulation_years = simulation_days / 365
    
    cycle_degradation = battery.calculate_cycle_degradation(total_energy_throughput)
    calendar_degradation = battery.calculate_calendar_degradation(simulation_years)
    total_degradation = cycle_degradation + calendar_degradation
    
    # Bereken resterende levensduur
    remaining_cycles = battery.calculate_remaining_cycles(total_degradation)
    
    if simulation_days > 0 and total_energy_throughput > 0:
        daily_throughput = total_energy_throughput / simulation_days
        daily_cycles = daily_throughput / (battery.capacity * 1000 * 2)  # Delen door 2 omdat een cyclus = laden + ontladen
        remaining_days = remaining_cycles / daily_cycles if daily_cycles > 0 else float('inf')
        remaining_years = remaining_days / 365
    else:
        daily_throughput = 0.0
        daily_cycles = 0.0
        remaining_days = float('inf')
        remaining_years = float('inf')
    
    # Stel resultaten samen
    results = {
        'simulation_data': df,
        'total_energy_to_battery': total_energy_to_battery,
        'total_energy_from_battery': total_energy_from_battery,
        'total_self_discharge': total_self_discharge,
        'total_extra_consumption': total_extra_consumption,
        'total_grid_import': total_grid_import,
        'total_grid_export': total_grid_export,
        'round_trip_efficiency': round_trip_efficiency,
        'original_import_cost': original_import_cost,
        'original_export_revenue': original_export_revenue,
        'original_net_cost': original_net_cost,
        'new_import_cost': new_import_cost,
        'new_export_revenue': new_export_revenue,
        'new_net_cost': new_net_cost,
        'savings': savings,
        'original_self_consumption': original_self_consumption,
        'new_self_consumption': new_self_consumption,
        'self_consumption_increase': self_consumption_increase,
        'cycle_degradation': cycle_degradation,
        'calendar_degradation': calendar_degradation,
        'total_degradation': total_degradation,
        'daily_throughput': daily_throughput,
        'daily_cycles': daily_cycles,
        'remaining_cycles': remaining_cycles,
        'remaining_days': remaining_days,
        'remaining_years': remaining_years,
        'battery_model': battery,
        'max_energy_content': max_energy_content,
        'control_strategy': control_strategy
    }
    
    return results


def create_battery_simulation_plots(simulation_results: Dict[str, Any]) -> Dict[str, go.Figure]:
    """Creëer visualisaties van de accusimulatie.
    
    Args:
        simulation_results: Resultaten van de accusimulatie
        
    Returns:
        Dict[str, go.Figure]: Dictionary met Plotly figuren
    """
    df = simulation_results['simulation_data']
    battery = simulation_results['battery_model']
    max_energy_content = simulation_results['max_energy_content']
    
    figures = {}
    
    # 1. State of Charge en energieinhoud plot
    fig_soc = go.Figure()
    
    # State of Charge
    fig_soc.add_trace(go.Scatter(
        x=df.index,
        y=df['Battery SoC'] * 100,  # Converteer naar percentage
        mode='lines',
        name='State of Charge',
        line=dict(color='blue')
    ))
    
    # Voeg horizontale lijnen toe voor min en max SoC
    fig_soc.add_shape(
        type="line",
        x0=df.index[0],
        y0=battery.min_soc * 100,
        x1=df.index[-1],
        y1=battery.min_soc * 100,
        line=dict(color="red", width=1, dash="dash"),
    )
    
    fig_soc.add_shape(
        type="line",
        x0=df.index[0],
        y0=battery.max_soc * 100,
        x1=df.index[-1],
        y1=battery.max_soc * 100,
        line=dict(color="green", width=1, dash="dash"),
    )
    
    # Secundaire y-as voor energieinhoud
    fig_soc.add_trace(go.Scatter(
        x=df.index,
        y=df['Battery Energy Content (Wh)'],
        mode='lines',
        name='Energieinhoud',
        line=dict(color='orange'),
        yaxis="y2"
    ))
    
    fig_soc.update_layout(
        title="Accu State of Charge en Energieinhoud",
        xaxis_title="Datum/Tijd",
        yaxis_title="State of Charge (%)",
        yaxis2=dict(
            title="Energieinhoud (Wh)",
            overlaying="y",
            side="right",
            range=[0, max_energy_content * 1.1]
        ),
        hovermode="x unified"
    )
    
    figures['soc_energy'] = fig_soc
    
    # 2. Energiestromen plot
    fig_energy = go.Figure()
    
    # Energie naar accu
    fig_energy.add_trace(go.Scatter(
        x=df.index,
        y=df['Energy to Battery (Wh)'],
        mode='lines',
        name='Energie naar Accu',
        line=dict(color='green')
    ))
    
    # Energie uit accu
    fig_energy.add_trace(go.Scatter(
        x=df.index,
        y=df['Energy from Battery (Wh)'],
        mode='lines',
        name='Energie uit Accu',
        line=dict(color='red')
    ))
    
    # Zelfontlading
    fig_energy.add_trace(go.Scatter(
        x=df.index,
        y=df['Self Discharge (Wh)'],
        mode='lines',
        name='Zelfontlading',
        line=dict(color='gray')
    ))
    
    fig_energy.update_layout(
        title="Energiestromen Accu",
        xaxis_title="Datum/Tijd",
        yaxis_title="Energie (Wh)",
        hovermode="x unified"
    )
    
    figures['energy_flows'] = fig_energy
    
    # 3. Grid interactie plot
    fig_grid = go.Figure()
    
    # Grid import
    fig_grid.add_trace(go.Scatter(
        x=df.index,
        y=df['Grid Import (Wh)'],
        mode='lines',
        name='Import van Net',
        line=dict(color='red')
    ))
    
    # Grid export
    fig_grid.add_trace(go.Scatter(
        x=df.index,
        y=df['Grid Export (Wh)'],
        mode='lines',
        name='Export naar Net',
        line=dict(color='green')
    ))
    
    # Netto energie zonder accu
    fig_grid.add_trace(go.Scatter(
        x=df.index,
        y=df['Net Energy (Wh)'].clip(lower=0),  # Alleen positieve waarden (export)
        mode='lines',
        name='Export zonder Accu',
        line=dict(color='lightgreen', dash='dash')
    ))
    
    fig_grid.add_trace(go.Scatter(
        x=df.index,
        y=df['Net Energy (Wh)'].clip(upper=0) * -1,  # Alleen negatieve waarden (import), maak positief
        mode='lines',
        name='Import zonder Accu',
        line=dict(color='lightcoral', dash='dash')
    ))
    
    fig_grid.update_layout(
        title="Interactie met Elektriciteitsnet",
        xaxis_title="Datum/Tijd",
        yaxis_title="Energie (Wh)",
        hovermode="x unified"
    )
    
    figures['grid_interaction'] = fig_grid
    
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
        y=df['Energy to Battery (Wh)'],
        mode='lines',
        name='Benut voor Accu',
        line=dict(color='darkgreen')
    ))
    
    fig_comparison.update_layout(
        title="Benutting van Energieoverschot",
        xaxis_title="Datum/Tijd",
        yaxis_title="Energie (Wh)",
        hovermode="x unified"
    )
    
    figures['surplus_utilization'] = fig_comparison
    
    # 5. Dagelijks patroon plot
    # Voeg uurkolom toe voor groepering
    df_hourly = df.copy()
    df_hourly['hour'] = df_hourly.index.hour
    
    # Bereken gemiddelden per uur
    hourly_avg = df_hourly.groupby('hour').agg({
        'Energy to Battery (Wh)': 'mean',
        'Energy from Battery (Wh)': 'mean',
        'Grid Import (Wh)': 'mean',
        'Grid Export (Wh)': 'mean',
        'Net Energy (Wh)': 'mean'
    }).reset_index()
    
    fig_daily = go.Figure()
    
    # Energie naar accu
    fig_daily.add_trace(go.Scatter(
        x=hourly_avg['hour'],
        y=hourly_avg['Energy to Battery (Wh)'],
        mode='lines+markers',
        name='Gem. Laden',
        line=dict(color='green')
    ))
    
    # Energie uit accu
    fig_daily.add_trace(go.Scatter(
        x=hourly_avg['hour'],
        y=hourly_avg['Energy from Battery (Wh)'],
        mode='lines+markers',
        name='Gem. Ontladen',
        line=dict(color='red')
    ))
    
    # Netto energie
    fig_daily.add_trace(go.Scatter(
        x=hourly_avg['hour'],
        y=hourly_avg['Net Energy (Wh)'],
        mode='lines+markers',
        name='Gem. Netto Energie',
        line=dict(color='blue', dash='dash')
    ))
    
    fig_daily.update_layout(
        title="Gemiddeld Dagelijks Patroon",
        xaxis_title="Uur van de dag",
        yaxis_title="Gemiddelde Energie (Wh)",
        xaxis=dict(tickmode='linear', tick0=0, dtick=1),
        hovermode="x unified"
    )
    
    figures['daily_pattern'] = fig_daily
    
    return figures


def calculate_payback_period(simulation_results: Dict[str, Any], 
                            battery_cost: float, 
                            installation_cost: float = 0,
                            annual_maintenance: float = 0,
                            replacement_years: Optional[int] = None,
                            replacement_cost_factor: float = 0.7,
                            years: int = 15,
                            discount_rate: float = 0.03) -> Dict[str, Any]:
    """Bereken de terugverdientijd en financiële analyse voor een accu-investering.
    
    Args:
        simulation_results: Resultaten van de accusimulatie
        battery_cost: Aanschafkosten van de accu in €
        installation_cost: Installatiekosten in €
        annual_maintenance: Jaarlijkse onderhoudskosten in €
        replacement_years: Aantal jaren na hoeveel de accu vervangen moet worden (None = geen vervanging)
        replacement_cost_factor: Factor voor de kosten van vervanging t.o.v. originele kosten
        years: Aantal jaren voor de analyse
        discount_rate: Discontovoet voor NCW berekening
        
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
    total_investment = battery_cost + installation_cost
    
    # Bereken cumulatieve kasstromen over de jaren
    cash_flows = []
    cumulative_cash_flow = -total_investment
    
    for year in range(1, years + 1):
        yearly_cash_flow = annual_savings - annual_maintenance
        
        # Voeg vervangingskosten toe indien nodig
        replacement_cost = 0
        if replacement_years is not None and year > 0 and year % replacement_years == 0:
            replacement_cost = battery_cost * replacement_cost_factor
            yearly_cash_flow -= replacement_cost
        
        cumulative_cash_flow += yearly_cash_flow
        cash_flows.append({
            'year': year,
            'yearly_savings': annual_savings,
            'yearly_maintenance': annual_maintenance,
            'replacement_cost': replacement_cost,
            'yearly_cash_flow': yearly_cash_flow,
            'cumulative_cash_flow': cumulative_cash_flow
        })
    
    # Bereken terugverdientijd in jaren
    if annual_savings <= annual_maintenance:
        payback_period = float('inf')  # Nooit terugverdiend
    else:
        # Eenvoudige terugverdientijd zonder rekening te houden met vervangingen
        simple_payback_period = total_investment / (annual_savings - annual_maintenance)
        
        # Bepaal werkelijke terugverdientijd met vervangingen
        if replacement_years is not None and simple_payback_period > replacement_years:
            # Als de terugverdientijd langer is dan de vervangingsperiode, moeten we de kasstromen analyseren
            payback_period = float('inf')
            for i, cf in enumerate(cash_flows):
                if cf['cumulative_cash_flow'] >= 0:
                    # Lineaire interpolatie voor nauwkeurigere schatting
                    if i > 0:
                        prev_cf = cash_flows[i-1]
                        fraction = -prev_cf['cumulative_cash_flow'] / (cf['cumulative_cash_flow'] - prev_cf['cumulative_cash_flow'])
                        payback_period = prev_cf['year'] + fraction
                    else:
                        payback_period = cf['year']
                    break
        else:
            payback_period = simple_payback_period
    
    # Bereken netto contante waarde (NCW)
    npv = -total_investment
    
    for year in range(1, years + 1):
        yearly_cash_flow = annual_savings - annual_maintenance
        
        # Voeg vervangingskosten toe indien nodig
        if replacement_years is not None and year > 0 and year % replacement_years == 0:
            replacement_cost = battery_cost * replacement_cost_factor
            yearly_cash_flow -= replacement_cost
        
        npv += yearly_cash_flow / ((1 + discount_rate) ** year)
    
    # Bereken return on investment (ROI) na de geanalyseerde periode
    if total_investment > 0:
        roi = (cumulative_cash_flow + total_investment) / total_investment * 100
    else:
        roi = 0.0
    
    # Bereken interne rentevoet (IRR)
    cash_flow_values = [-total_investment]
    for cf in cash_flows:
        cash_flow_values.append(cf['yearly_cash_flow'])
    
    try:
        irr = np.irr(cash_flow_values)
    except:
        irr = None  # IRR kan niet worden berekend (bijv. als alle waarden negatief zijn)
    
    # Stel resultaten samen
    results = {
        'annual_savings': annual_savings,
        'total_investment': total_investment,
        'payback_period': payback_period,
        'npv': npv,
        'roi': roi,
        'irr': irr,
        'cash_flows': cash_flows,
        'simulation_days': simulation_days,
        'replacement_years': replacement_years
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
    
    # Voeg jaarlijkse kasstromen toe als staafdiagram
    fig.add_trace(go.Bar(
        x=df['year'],
        y=df['yearly_cash_flow'],
        name='Jaarlijkse Kasstroom',
        marker_color='lightblue',
        opacity=0.7,
        yaxis='y2'
    ))
    
    # Markeer vervangingsmomenten indien van toepassing
    if financial_results['replacement_years'] is not None:
        replacement_years = []
        for year in range(financial_results['replacement_years'], df['year'].max() + 1, financial_results['replacement_years']):
            replacement_years.append(year)
        
        for year in replacement_years:
            fig.add_shape(
                type="line",
                x0=year,
                y0=df['cumulative_cash_flow'].min(),
                x1=year,
                y1=df['cumulative_cash_flow'].max(),
                line=dict(color="red", width=1, dash="dash"),
            )
            
            fig.add_annotation(
                x=year,
                y=df['cumulative_cash_flow'].min(),
                text=f"Vervanging",
                showarrow=True,
                arrowhead=1,
                yshift=10
            )
    
    # Voeg horizontale lijn toe bij y=0 (terugverdienmoment)
    fig.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=df['year'].max(),
        y1=0,
        line=dict(color="green", width=1, dash="dash"),
    )
    
    # Voeg verticale lijn toe bij terugverdientijd
    if financial_results['payback_period'] <= df['year'].max() and financial_results['payback_period'] != float('inf'):
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
        title="Financiële Analyse Accu",
        xaxis_title="Jaar",
        yaxis_title="Cumulatieve Kasstroom (€)",
        yaxis2=dict(
            title="Jaarlijkse Kasstroom (€)",
            overlaying="y",
            side="right"
        ),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


def optimize_battery_parameters(data: pd.DataFrame, 
                              capacity_range: List[float] = [5.0, 7.5, 10.0, 15.0],
                              control_strategies: List[str] = ["maximize_self_consumption", "time_of_use", "peak_shaving"]) -> Dict[str, Any]:
    """Optimaliseer accuparameters voor maximale besparingen.
    
    Args:
        data: DataFrame met energiedata
        capacity_range: Lijst met te evalueren accucapaciteiten in kWh
        control_strategies: Lijst met te evalueren besturingsstrategieën
        
    Returns:
        Dict[str, Any]: Resultaten van de optimalisatie
    """
    results = []
    
    for capacity in capacity_range:
        for strategy in control_strategies:
            # Maak een accumodel met de huidige capaciteit
            battery = BatteryModel(capacity=capacity)
            
            # Simuleer met de huidige strategie
            simulation = simulate_battery_storage(
                data=data,
                battery=battery,
                control_strategy=strategy
            )
            
            # Bereken financiële resultaten (aanname: €500 per kWh)
            battery_cost = 500 * capacity
            financial = calculate_payback_period(
                simulation_results=simulation,
                battery_cost=battery_cost,
                replacement_years=10  # Aanname: vervanging na 10 jaar
            )
            
            # Sla resultaten op
            results.append({
                'capacity': capacity,
                'strategy': strategy,
                'annual_savings': financial['annual_savings'],
                'payback_period': financial['payback_period'],
                'npv': financial['npv'],
                'roi': financial['roi'],
                'battery_cost': battery_cost,
                'total_energy_to_battery': simulation['total_energy_to_battery'],
                'total_energy_from_battery': simulation['total_energy_from_battery'],
                'round_trip_efficiency': simulation['round_trip_efficiency'],
                'self_consumption_increase': simulation['self_consumption_increase'],
                'remaining_years': simulation['remaining_years']
            })
    
    # Converteer naar DataFrame voor eenvoudigere analyse
    results_df = pd.DataFrame(results)
    
    # Vind de optimale configuratie (hoogste NPV)
    if len(results_df) > 0:
        optimal_config = results_df.loc[results_df['npv'].idxmax()]
    else:
        optimal_config = None
    
    # Stel resultaten samen
    optimization_results = {
        'all_results': results_df,
        'optimal_config': optimal_config,
        'capacity_range': capacity_range,
        'control_strategies': control_strategies
    }
    
    return optimization_results


def create_optimization_plots(optimization_results: Dict[str, Any]) -> Dict[str, go.Figure]:
    """Creëer visualisaties van de optimalisatieresultaten.
    
    Args:
        optimization_results: Resultaten van de accu-optimalisatie
        
    Returns:
        Dict[str, go.Figure]: Dictionary met Plotly figuren
    """
    df = optimization_results['all_results']
    
    figures = {}
    
    # 1. Terugverdientijd per capaciteit en strategie
    fig_payback = px.bar(
        df,
        x='capacity',
        y='payback_period',
        color='strategy',
        barmode='group',
        title="Terugverdientijd per Accucapaciteit en Strategie",
        labels={
            'capacity': 'Accucapaciteit (kWh)',
            'payback_period': 'Terugverdientijd (jaren)',
            'strategy': 'Besturingsstrategie'
        }
    )
    
    # Voeg horizontale lijn toe bij 10 jaar (typische levensduur)
    fig_payback.add_shape(
        type="line",
        x0=df['capacity'].min() - 1,
        y0=10,
        x1=df['capacity'].max() + 1,
        y1=10,
        line=dict(color="red", width=1, dash="dash"),
    )
    
    fig_payback.add_annotation(
        x=df['capacity'].max(),
        y=10,
        text="Typische levensduur",
        showarrow=False,
        yshift=10
    )
    
    figures['payback_comparison'] = fig_payback
    
    # 2. Jaarlijkse besparingen per capaciteit en strategie
    fig_savings = px.bar(
        df,
        x='capacity',
        y='annual_savings',
        color='strategy',
        barmode='group',
        title="Jaarlijkse Besparingen per Accucapaciteit en Strategie",
        labels={
            'capacity': 'Accucapaciteit (kWh)',
            'annual_savings': 'Jaarlijkse Besparingen (€)',
            'strategy': 'Besturingsstrategie'
        }
    )
    
    figures['savings_comparison'] = fig_savings
    
    # 3. Netto Contante Waarde (NPV) per capaciteit en strategie
    fig_npv = px.bar(
        df,
        x='capacity',
        y='npv',
        color='strategy',
        barmode='group',
        title="Netto Contante Waarde per Accucapaciteit en Strategie",
        labels={
            'capacity': 'Accucapaciteit (kWh)',
            'npv': 'Netto Contante Waarde (€)',
            'strategy': 'Besturingsstrategie'
        }
    )
    
    # Voeg horizontale lijn toe bij y=0 (break-even)
    fig_npv.add_shape(
        type="line",
        x0=df['capacity'].min() - 1,
        y0=0,
        x1=df['capacity'].max() + 1,
        y1=0,
        line=dict(color="green", width=1, dash="dash"),
    )
    
    figures['npv_comparison'] = fig_npv
    
    # 4. Efficiëntie en zelfconsumptie per capaciteit en strategie
    fig_efficiency = px.line(
        df,
        x='capacity',
        y='round_trip_efficiency',
        color='strategy',
        markers=True,
        title="Efficiëntie en Zelfconsumptie per Accucapaciteit en Strategie",
        labels={
            'capacity': 'Accucapaciteit (kWh)',
            'round_trip_efficiency': 'Round-trip Efficiëntie (%)',
            'strategy': 'Besturingsstrategie'
        }
    )
    
    # Voeg zelfconsumptie toe op secundaire y-as
    fig_self_consumption = px.line(
        df,
        x='capacity',
        y='self_consumption_increase',
        color='strategy',
        markers=True,
        labels={
            'capacity': 'Accucapaciteit (kWh)',
            'self_consumption_increase': 'Toename Zelfconsumptie (%)',
            'strategy': 'Besturingsstrategie'
        }
    )
    
    # Combineer de twee figuren
    for trace in fig_self_consumption.data:
        trace.update(yaxis="y2")
        trace.update(line=dict(dash="dash"))
        fig_efficiency.add_trace(trace)
    
    fig_efficiency.update_layout(
        yaxis2=dict(
            title="Toename Zelfconsumptie (%)",
            overlaying="y",
            side="right"
        ),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    figures['efficiency_comparison'] = fig_efficiency
    
    return figures