"""Tests voor de battery_module."""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from storage_modules.battery_module import BatteryModel, simulate_battery_storage


class TestBatteryModel(unittest.TestCase):
    """Test cases voor de BatteryModel klasse."""
    
    def setUp(self):
        """Initialiseer een standaard BatteryModel voor tests."""
        self.battery = BatteryModel(
            capacity=10.0,  # kWh
            max_charge_rate=3.7,  # kW
            max_discharge_rate=3.7,  # kW
            charge_efficiency=0.95,
            discharge_efficiency=0.95,
            self_discharge_rate=0.002,  # 0.2% per dag
            min_soc=0.1,
            max_soc=0.9
        )
    
    def test_init(self):
        """Test of de BatteryModel correct wordt geïnitialiseerd."""
        self.assertEqual(self.battery.capacity, 10.0)
        self.assertEqual(self.battery.max_charge_rate, 3.7)
        self.assertEqual(self.battery.max_discharge_rate, 3.7)
        self.assertEqual(self.battery.charge_efficiency, 0.95)
        self.assertEqual(self.battery.discharge_efficiency, 0.95)
        self.assertEqual(self.battery.self_discharge_rate, 0.002)
        self.assertEqual(self.battery.min_soc, 0.1)
        self.assertEqual(self.battery.max_soc, 0.9)
        
        # Test berekende waarden
        self.assertEqual(self.battery.usable_capacity, 8.0)  # 10.0 * (0.9 - 0.1)
    
    def test_calculate_storage_capacity(self):
        """Test of de opslagcapaciteit correct wordt berekend."""
        capacity = self.battery.calculate_storage_capacity()
        self.assertEqual(capacity, 8000.0)  # 8.0 kWh = 8000 Wh
    
    def test_calculate_self_discharge(self):
        """Test of de zelfontlading correct wordt berekend."""
        # Test voor 1 dag
        discharge = self.battery.calculate_self_discharge(1000.0, 1.0)
        self.assertAlmostEqual(discharge, 2.0)  # 0.2% van 1000 Wh = 2 Wh
        
        # Test voor 10 dagen
        discharge = self.battery.calculate_self_discharge(1000.0, 10.0)
        expected = 1000.0 * (1 - (1 - 0.002) ** 10)
        self.assertAlmostEqual(discharge, expected)
    
    def test_calculate_cycle_degradation(self):
        """Test of de cyclusdegradatie correct wordt berekend."""
        # Test voor 1 volledige cyclus (laden + ontladen = 2 * capaciteit)
        degradation = self.battery.calculate_cycle_degradation(2 * 10000.0)
        self.assertEqual(degradation, 0.0005)  # 0.05% per cyclus
        
        # Test voor 10 volledige cycli
        degradation = self.battery.calculate_cycle_degradation(10 * 2 * 10000.0)
        self.assertEqual(degradation, 0.005)  # 0.5% voor 10 cycli
    
    def test_calculate_calendar_degradation(self):
        """Test of de kalenderdegradatie correct wordt berekend."""
        # Test voor 1 jaar
        degradation = self.battery.calculate_calendar_degradation(1.0)
        self.assertEqual(degradation, 0.02)  # 2% per jaar
        
        # Test voor 5 jaar
        degradation = self.battery.calculate_calendar_degradation(5.0)
        self.assertEqual(degradation, 0.1)  # 10% voor 5 jaar


class TestBatterySimulation(unittest.TestCase):
    """Test cases voor de battery_module simulatiefuncties."""
    
    def setUp(self):
        """Initialiseer testdata en een BatteryModel voor simulatietests."""
        # Maak een eenvoudige dataset met een DatetimeIndex
        start_date = datetime(2023, 1, 1)
        dates = [start_date + timedelta(hours=i) for i in range(24)]  # 24 uur aan data
        
        # Maak een patroon van energieproductie en -verbruik
        production = [0, 0, 0, 0, 0, 0, 100, 300, 500, 700, 800, 900, 
                      1000, 900, 800, 700, 500, 300, 100, 0, 0, 0, 0, 0]  # Wh
        consumption = [200, 150, 100, 100, 100, 150, 200, 300, 400, 300, 200, 200, 
                       300, 300, 200, 300, 400, 500, 600, 500, 400, 300, 250, 200]  # Wh
        
        # Bereken netto energie (productie - verbruik)
        net_energy = [p - c for p, c in zip(production, consumption)]
        
        # Maak een DataFrame
        self.data = pd.DataFrame({
            'Energy Produced (Wh)': production,
            'Energy Consumed (Wh)': consumption,
            'Net Energy (Wh)': net_energy
        }, index=dates)
        
        # Maak een BatteryModel
        self.battery = BatteryModel(
            capacity=5.0,  # kWh
            max_charge_rate=2.0,  # kW
            max_discharge_rate=2.0,  # kW
            charge_efficiency=0.95,
            discharge_efficiency=0.95
        )
    
    def test_simulate_battery_storage(self):
        """Test of de accusimulatie correct werkt."""
        # Voer de simulatie uit
        results = simulate_battery_storage(
            data=self.data,
            battery=self.battery,
            control_strategy="maximize_self_consumption"
        )
        
        # Controleer of de resultaten de verwachte keys bevatten
        expected_keys = [
            'simulation_data', 'total_energy_to_battery', 'total_energy_from_battery',
            'total_self_discharge', 'round_trip_efficiency', 'savings'
        ]
        for key in expected_keys:
            self.assertIn(key, results)
        
        # Controleer of de simulatiedata de verwachte kolommen bevat
        expected_columns = [
            'Battery SoC', 'Battery Energy Content (Wh)', 'Energy to Battery (Wh)',
            'Energy from Battery (Wh)', 'Self Discharge (Wh)', 'Grid Import (Wh)',
            'Grid Export (Wh)'
        ]
        for col in expected_columns:
            self.assertIn(col, results['simulation_data'].columns)
        
        # Controleer of de totalen niet-negatief zijn
        self.assertGreaterEqual(results['total_energy_to_battery'], 0)
        self.assertGreaterEqual(results['total_energy_from_battery'], 0)
        self.assertGreaterEqual(results['total_self_discharge'], 0)
        
        # Controleer of de round-trip efficiëntie tussen 0 en 100% ligt
        if results['total_energy_to_battery'] > 0:
            self.assertGreaterEqual(results['round_trip_efficiency'], 0)
            self.assertLessEqual(results['round_trip_efficiency'], 100)


if __name__ == '__main__':
    unittest.main()