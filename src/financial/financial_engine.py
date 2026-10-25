class FinancialEngine:
    """
    Calculates daily monetary loss and repair economics.
    Converts physical energy loss into operational financial metrics.
    """
    def __init__(self, default_tariff=4.50, default_repair_cost=40000.0):
        self.default_tariff = default_tariff
        self.default_repair_cost = default_repair_cost

    def calculate_losses(self, daily_energy_loss_kwh, tariff=None, repair_cost=None, capacity_factor=0.36):
        # Validate inputs against physical boundaries
        daily_kwh = max(0.0, float(daily_energy_loss_kwh))
        t = max(0.0, float(tariff if tariff is not None else self.default_tariff))
        rc = max(0.0, float(repair_cost if repair_cost is not None else self.default_repair_cost))
        cf = max(0.05, min(1.0, float(capacity_factor if capacity_factor is not None else 0.36)))

        daily_loss_inr = round(daily_kwh * t, 2)
        annual_loss_inr = round(daily_loss_inr * 365.0, 2)
        monthly_loss_inr = round(daily_loss_inr * 30.0, 2)

        if daily_loss_inr > 0.0:
            payback_days = round(rc / daily_loss_inr, 1)
            payback_years = round(payback_days / 365.0, 2)
            # Sensitivity bounds (+/- 15% tariff volatility and +/- 10% capacity factor variation)
            daily_loss_lower = round(daily_kwh * (t * 0.85) * (0.90), 2)
            daily_loss_upper = round(daily_kwh * (t * 1.15) * (1.10), 2)
        else:
            payback_days = None
            payback_years = None
            daily_loss_lower = 0.0
            daily_loss_upper = 0.0

        return {
            'tariff_inr_kwh': t,
            'repair_cost_inr': rc,
            'capacity_factor': cf,
            'daily_loss_inr': daily_loss_inr,
            'monthly_loss_inr': monthly_loss_inr,
            'annual_loss_inr': annual_loss_inr,
            'payback_days': payback_days,
            'payback_years': payback_years,
            'sensitivity_range_inr': {
                'daily_loss_p10': daily_loss_lower,
                'daily_loss_p90': daily_loss_upper
            },
            'financial_assumptions': {
                'annual_operating_days': 365,
                'continuous_production_proxy': 'Baseline daily energy derived from annualized capacity factor (CF)',
                'currency': 'INR'
            }
        }
