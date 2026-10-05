import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.financial.financial_engine import FinancialEngine

class FinancialService:
    """
    Financial Loss & Repair Economics Service.
    Calculates turbine-specific daily, monthly, and annual revenue losses,
    repair payback period, and economic trade-offs.
    """
    def __init__(self):
        self.engine = FinancialEngine()

    def compute_financials(self, daily_energy_loss_kwh, tariff=4.50, repair_cost=40000.0):
        res = self.engine.calculate_losses(
            daily_energy_loss_kwh=daily_energy_loss_kwh,
            tariff=tariff,
            repair_cost=repair_cost
        )
        res['daily_financial_loss_inr'] = res['daily_loss_inr']
        res['monthly_financial_loss_inr'] = res['monthly_loss_inr']
        res['annual_financial_loss_inr'] = res['annual_loss_inr']
        return res
