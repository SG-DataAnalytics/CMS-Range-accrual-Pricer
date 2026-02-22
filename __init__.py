"""
CMS Range Accrual Pricer
=========================

A professional Python toolkit for pricing CMS Range Accrual products
with comprehensive risk analytics and visualization.

Built for rates trading desks.

Quick Start:
-----------
>>> from cms_pricer import CurveBuilder, ProductFactory, MonteCarloEngine
>>> from cms_pricer import CMSRateSimulator, ModelParameters, RiskAnalyzer
>>> 
>>> # Build curves
>>> ois_curve, swap_curve = CurveBuilder.build_sample_curves()
>>> 
>>> # Define product
>>> product = ProductFactory.create_standard_range_accrual()
>>> 
>>> # Setup model and price
>>> params = ModelParameters()
>>> simulator = CMSRateSimulator(initial_rate=0.032, parameters=params)
>>> mc_engine = MonteCarloEngine(discount_curve=ois_curve)
>>> 
>>> result = mc_engine.price_product(
>>>     simulator=simulator,
>>>     payoff_function=product.payoff_function(),
>>>     maturity=product.maturity
>>> )
>>> print(f"Price: ${result['price']:,.2f}")

Modules:
--------
- yield_curve: Curve construction and interpolation
- monte_carlo: MC engine and CMS rate simulation
- products: Range accrual product definitions
- risk_analytics: DV01, slope risk, sensitivity analysis
- visualization: Professional plotting tools
"""

__version__ = "1.0.0"
__author__ = "Rates Trading Desk"
__all__ = [
    # Yield Curve
    'YieldCurve',
    'CurveBuilder',
    'MarketQuote',
    'InterpolationMethod',
    
    # Monte Carlo
    'MonteCarloEngine',
    'CMSRateSimulator',
    'ModelParameters',
    
    # Products
    'RangeAccrualProduct',
    'DigitalRangeAccrual',
    'KnockOutRangeAccrual',
    'ProductFactory',
    'DayCountConvention',
    
    # Risk Analytics
    'RiskAnalyzer',
    'RiskMetrics',
    
    # Visualization
    'Visualizer',
]

# Import main classes for convenience
from .yield_curve import (
    YieldCurve,
    CurveBuilder,
    MarketQuote,
    InterpolationMethod
)

from .monte_carlo import (
    MonteCarloEngine,
    CMSRateSimulator,
    ModelParameters
)

from .products import (
    RangeAccrualProduct,
    DigitalRangeAccrual,
    KnockOutRangeAccrual,
    ProductFactory,
    DayCountConvention
)

from .risk_analytics import (
    RiskAnalyzer,
    RiskMetrics
)

from .visualization import Visualizer
