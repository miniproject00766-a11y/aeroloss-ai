class FinancialEngine:
    """
    Calculates daily monetary loss and repair economics.
    Converts physical energy loss into operational financial metrics.
    """
    def __init__(self, default_tariff=4.50, default_repair_cost=40000.0):
        self.default_tariff = default_tariff
        self.default_repair_cost = default_repair_cost

    def calculate_losses(self, daily_energy_loss_kwh, tariff=None, repair_cost=None):
        t = tariff if tariff is not None else self.default_tariff
        rc = repair_cost if repair_cost is not None else self.default_repair_cost

        daily_loss_inr = round(daily_energy_loss_kwh * t, 2)
        monthly_loss_inr = round(daily_loss_inr * 30.0, 2)
        annual_loss_inr = round(daily_loss_inr * 365.0, 2)

        if daily_loss_inr > 0:
            payback_days = round(rc / daily_loss_inr, 1)
        else:
            payback_days = 999.0

        return {
            'tariff_inr_kwh': t,
            'repair_cost_inr': rc,
            'daily_loss_inr': daily_loss_inr,
            'monthly_loss_inr': monthly_loss_inr,
            'annual_loss_inr': annual_loss_inr,
            'payback_days': payback_days
        }
