"""Tests voor de data_loader module."""

import os
import sys
import unittest
import pandas as pd
import numpy as np
from pathlib import Path

# Voeg de projectroot toe aan het pad zodat we de modules kunnen importeren
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_processing import data_loader


class TestDataLoader(unittest.TestCase):
    """Test cases voor de data_loader module."""
    
    def setUp(self):
        """Voorbereiding voor de tests."""
        # Maak een tijdelijk testbestand aan
        self.test_data_path = 'tests/test_data.csv'
        
        # Maak de tests directory als deze nog niet bestaat
        os.makedirs(os.path.dirname(self.test_data_path), exist_ok=True)
        
        # Maak een eenvoudige dataset
        data = {
            'Date/Time': ['01/01/2024 00:00', '01/01/2024 00:15', '01/01/2024 00:30'],
            'Energy Produced (Wh)': [0, 0, 10],
            'Energy Consumed (Wh)': [100, 90, 80],
            'Exported to Grid (Wh)': [0, 0, 0],
            'Imported from Grid (Wh)': [100, 90, 70]
        }
        pd.DataFrame(data).to_csv(self.test_data_path, index=False)
    
    def tearDown(self):
        """Opruimen na de tests."""
        # Verwijder het testbestand als het bestaat
        if os.path.exists(self.test_data_path):
            os.remove(self.test_data_path)
    
    def test_load_csv_data(self):
        """Test het laden van CSV-data."""
        df = data_loader.load_csv_data(self.test_data_path)
        
        # Controleer of het juiste aantal rijen is geladen
        self.assertEqual(len(df), 3)
        
        # Controleer of de kolommen correct zijn
        expected_columns = [
            'Energy Produced (Wh)', 
            'Energy Consumed (Wh)', 
            'Exported to Grid (Wh)', 
            'Imported from Grid (Wh)'
        ]
        for col in expected_columns:
            self.assertIn(col, df.columns)
        
        # Controleer of de index een DatetimeIndex is
        self.assertIsInstance(df.index, pd.DatetimeIndex)
    
    def test_validate_data(self):
        """Test de datavalidatie."""
        # Laad de testdata
        df = data_loader.load_csv_data(self.test_data_path)
        
        # Test met geldige data
        is_valid, errors = data_loader.validate_data(df)
        self.assertFalse(is_valid)  # Onze testdata is niet in balans
        
        # Maak een gebalanceerde dataset
        balanced_df = df.copy()
        balanced_df['Exported to Grid (Wh)'] = 0
        balanced_df['Imported from Grid (Wh)'] = balanced_df['Energy Consumed (Wh)'] - balanced_df['Energy Produced (Wh)']
        
        is_valid, errors = data_loader.validate_data(balanced_df)
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
        
        # Test met ongeldige data (negatieve waarden)
        invalid_df = df.copy()
        invalid_df['Energy Produced (Wh)'] = -10
        
        is_valid, errors = data_loader.validate_data(invalid_df)
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)
    
    def test_resample_data(self):
        """Test het resampling van data."""
        # Laad de testdata
        df = data_loader.load_csv_data(self.test_data_path)
        
        # Resample naar uurlijkse intervallen
        resampled = data_loader.resample_data(df, interval='1H')
        
        # Controleer of het aantal rijen is verminderd
        self.assertLess(len(resampled), len(df))
    
    def test_calculate_derived_values(self):
        """Test het berekenen van afgeleide waarden."""
        # Laad de testdata
        df = data_loader.load_csv_data(self.test_data_path)
        
        # Bereken afgeleide waarden
        derived = data_loader.calculate_derived_values(df)
        
        # Controleer of de nieuwe kolommen zijn toegevoegd
        self.assertIn('Net Energy (Wh)', derived.columns)
        self.assertIn('Self Consumption (Wh)', derived.columns)
        self.assertIn('Self Sufficiency (%)', derived.columns)
        
        # Controleer een specifieke berekening
        self.assertEqual(
            derived['Net Energy (Wh)'].iloc[0], 
            df['Energy Produced (Wh)'].iloc[0] - df['Energy Consumed (Wh)'].iloc[0]
        )
    
    def test_calculate_basic_statistics(self):
        """Test het berekenen van basisstatistieken."""
        # Laad de testdata
        df = data_loader.load_csv_data(self.test_data_path)
        
        # Bereken statistieken
        stats = data_loader.calculate_basic_statistics(df)
        
        # Controleer of de verwachte statistieken aanwezig zijn
        self.assertIn('Total Energy Produced (Wh)', stats)
        self.assertIn('Total Energy Consumed (Wh)', stats)
        
        # Controleer een specifieke berekening
        self.assertEqual(stats['Total Energy Produced (Wh)'], df['Energy Produced (Wh)'].sum())
    
    def test_create_sample_data(self):
        """Test het genereren van voorbeelddata."""
        sample_path = 'tests/sample_data.csv'
        
        # Genereer voorbeelddata
        data_loader.create_sample_data(sample_path, days=1)
        
        # Controleer of het bestand is aangemaakt
        self.assertTrue(os.path.exists(sample_path))
        
        # Laad en controleer de gegenereerde data
        df = pd.read_csv(sample_path)
        
        # Controleer of alle verwachte kolommen aanwezig zijn
        expected_columns = [
            'Date/Time',
            'Energy Produced (Wh)', 
            'Energy Consumed (Wh)', 
            'Exported to Grid (Wh)', 
            'Imported from Grid (Wh)'
        ]
        for col in expected_columns:
            self.assertIn(col, df.columns)
        
        # Controleer of er 24 uur * 4 kwartieren = 96 rijen zijn
        self.assertEqual(len(df), 96)
        
        # Ruim op
        if os.path.exists(sample_path):
            os.remove(sample_path)


if __name__ == '__main__':
    unittest.main()