

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from yield_curve import YieldCurve, MarketQuote
from monte_carlo import MonteCarloEngine, CMSRateSimulator
from products import RangeAccrualProduct


@dataclass
class RiskMetrics:
    dv01: float  # DV01 (price change per 1bp parallel shift)
    slope_risk: float  # Sensitivity to curve steepening
    gamma: float  # Convexity measure
    vega: Optional[float] = None  # Volatility sensitivity
    
    def __repr__(self) -> str:
        return (
            f"RiskMetrics(DV01={self.dv01:,.2f}, "
            f"Slope={self.slope_risk:,.2f}, "
            f"Gamma={self.gamma:,.2f})"
        )


class RiskAnalyzer:
    
    def __init__(
        self,
        product: RangeAccrualProduct,
        discount_curve: YieldCurve,
        simulator: CMSRateSimulator,
        mc_engine: MonteCarloEngine
    ):
        self.product = product
        self.discount_curve = discount_curve
        self.simulator = simulator
        self.mc_engine = mc_engine
    
    def calculate_dv01(
        self,
        bump_size: float = 0.0001,
        n_steps: int = 252
    ) -> Dict[str, float]:

        payoff_fn = self.product.payoff_function()
  
        base_result = self.mc_engine.price_product(
            self.simulator,
            payoff_fn,
            self.product.maturity,
            n_steps
        )
        base_price = base_result['price']
 
        up_curve = self.discount_curve.bump_curve(bump_size)
        up_engine = MonteCarloEngine(
            discount_curve=up_curve,
            n_simulations=self.mc_engine.n_simulations,
            random_seed=self.mc_engine.random_seed
        )
        up_result = up_engine.price_product(
            self.simulator,
            payoff_fn,
            self.product.maturity,
            n_steps
        )
        up_price = up_result['price']
        
        down_curve = self.discount_curve.bump_curve(-bump_size)
        down_engine = MonteCarloEngine(
            discount_curve=down_curve,
            n_simulations=self.mc_engine.n_simulations,
            random_seed=self.mc_engine.random_seed
        )
        down_result = down_engine.price_product(
            self.simulator,
            payoff_fn,
            self.product.maturity,
            n_steps
        )
        down_price = down_result['price']
        
        # Calculate metrics
        dv01 = (down_price - up_price) / (2 * bump_size)
        dv01_per_bp = dv01  # Already per bp
        dv01_per_million = dv01 / (self.product.notional / 1_000_000)
        
        return {
            'base_price': base_price,
            'up_price': up_price,
            'down_price': down_price,
            'dv01': dv01,
            'dv01_per_bp': dv01_per_bp,
            'dv01_per_million': dv01_per_million,
            'bump_size_bp': bump_size * 10000
        }
    
    def calculate_slope_risk(
        self,
        steepening_size: float = 0.0001,
        pivot_tenor: float = 5.0,
        n_steps: int = 252
    ) -> Dict[str, float]:

        payoff_fn = self.product.payoff_function()
        
        # Base case
        base_price = self.mc_engine.price_product(
            self.simulator,
            payoff_fn,
            self.product.maturity,
            n_steps
        )['price']
        
        # Create steepened curve
        steepened_curve = self._create_steepened_curve(
            self.discount_curve,
            steepening_size,
            pivot_tenor
        )
        
        steep_engine = MonteCarloEngine(
            discount_curve=steepened_curve,
            n_simulations=self.mc_engine.n_simulations,
            random_seed=self.mc_engine.random_seed
        )
        
        steep_price = steep_engine.price_product(
            self.simulator,
            payoff_fn,
            self.product.maturity,
            n_steps
        )['price']
    
        slope_risk = steep_price - base_price
        slope_risk_per_bp = slope_risk / steepening_size
        
        return {
            'base_price': base_price,
            'steepened_price': steep_price,
            'slope_risk': slope_risk,
            'slope_risk_per_bp': slope_risk_per_bp,
            'steepening_size_bp': steepening_size * 10000,
            'pivot_tenor': pivot_tenor
        }
    
    def _create_steepened_curve(
        self,
        base_curve: YieldCurve,
        steepening_size: float,
        pivot_tenor: float
    ) -> YieldCurve:

        new_quotes = []
        
        for tenor, rate in zip(base_curve.tenors, base_curve.rates):
            distance = tenor - pivot_tenor
            shift = steepening_size * distance / pivot_tenor
            new_rate = rate + shift
            
            new_quotes.append(MarketQuote(tenor=tenor, rate=new_rate))
        
        return YieldCurve(
            quotes=new_quotes,
            curve_name=f"{base_curve.curve_name}_steepened",
            interpolation=base_curve.interpolation
        )
    
    def calculate_gamma(
        self,
        bump_size: float = 0.0001,
        n_steps: int = 252
    ) -> Dict[str, float]:

        payoff_fn = self.product.payoff_function()
        
        dv01_results = self.calculate_dv01(bump_size, n_steps)
        
        base_price = dv01_results['base_price']
        up_price = dv01_results['up_price']
        down_price = dv01_results['down_price']
        
        gamma = (up_price + down_price - 2 * base_price) / (bump_size ** 2)
        
        convexity_pct = gamma / base_price * 100 if base_price != 0 else 0
        
        return {
            'gamma': gamma,
            'convexity_pct': convexity_pct,
            'base_price': base_price,
            'up_price': up_price,
            'down_price': down_price
        }
    
    def full_risk_report(self, n_steps: int = 252) -> RiskMetrics:

        dv01_result = self.calculate_dv01(n_steps=n_steps)
        slope_result = self.calculate_slope_risk(n_steps=n_steps)
        gamma_result = self.calculate_gamma(n_steps=n_steps)
        
        return RiskMetrics(
            dv01=dv01_result['dv01'],
            slope_risk=slope_result['slope_risk'],
            gamma=gamma_result['gamma']
        )
    
    def scenario_pnl(
        self,
        scenarios: List[Dict[str, float]],
        n_steps: int = 252
    ) -> Dict[str, Dict[str, float]]:
 
        payoff_fn = self.product.payoff_function()
        
        # Base price
        base_price = self.mc_engine.price_product(
            self.simulator,
            payoff_fn,
            self.product.maturity,
            n_steps
        )['price']
        
        results = {}
        
        for scenario in scenarios:
            name = scenario['name']
            shift = scenario['shift']
            
            shifted_curve = self.discount_curve.bump_curve(shift)
            shifted_engine = MonteCarloEngine(
                discount_curve=shifted_curve,
                n_simulations=self.mc_engine.n_simulations,
                random_seed=self.mc_engine.random_seed
            )
            
            shifted_price = shifted_engine.price_product(
                self.simulator,
                payoff_fn,
                self.product.maturity,
                n_steps
            )['price']
            
            pnl = shifted_price - base_price
            pnl_pct = (pnl / base_price * 100) if base_price != 0 else 0
            
            results[name] = {
                'price': shifted_price,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'shift_bp': shift * 10000
            }
        
        return results
    
    def ladder_risk(
        self,
        key_tenors: List[float],
        bump_size: float = 0.0001,
        n_steps: int = 252
    ) -> Dict[float, float]:

        payoff_fn = self.product.payoff_function()
        
        base_price = self.mc_engine.price_product(
            self.simulator,
            payoff_fn,
            self.product.maturity,
            n_steps
        )['price']
        
        ladder_risks = {}
        
        for tenor in key_tenors:
            bumped_curve = self._bump_single_tenor(
                self.discount_curve,
                tenor,
                bump_size
            )
            
            bumped_engine = MonteCarloEngine(
                discount_curve=bumped_curve,
                n_simulations=self.mc_engine.n_simulations,
                random_seed=self.mc_engine.random_seed
            )
            
            bumped_price = bumped_engine.price_product(
                self.simulator,
                payoff_fn,
                self.product.maturity,
                n_steps
            )['price']
            
            sensitivity = (bumped_price - base_price) / bump_size
            ladder_risks[tenor] = sensitivity
        
        return ladder_risks
    
    def _bump_single_tenor(
        self,
        curve: YieldCurve,
        target_tenor: float,
        bump_size: float
    ) -> YieldCurve:
        new_quotes = []
        
        for tenor, rate in zip(curve.tenors, curve.rates):
            if abs(tenor - target_tenor) < 0.01:  # Match within tolerance
                new_rate = rate + bump_size
            else:
                new_rate = rate
            
            new_quotes.append(MarketQuote(tenor=tenor, rate=new_rate))
        
        return YieldCurve(
            quotes=new_quotes,
            curve_name=f"{curve.curve_name}_bumped_{target_tenor}Y",
            interpolation=curve.interpolation
        )


if __name__ == "__main__":
    from yield_curve import CurveBuilder
    from monte_carlo import ModelParameters
    from products import ProductFactory

    ois_curve, _ = CurveBuilder.build_sample_curves()
    product = ProductFactory.create_standard_range_accrual()
    
    params = ModelParameters(mean_reversion=0.05, volatility=0.015)
    simulator = CMSRateSimulator(initial_rate=0.032, parameters=params)
    mc_engine = MonteCarloEngine(discount_curve=ois_curve, n_simulations=5000)
    
    analyzer = RiskAnalyzer(product, ois_curve, simulator, mc_engine)
    
    print("=" * 60)
    print("RISK ANALYSIS")
    print("=" * 60)
    
    dv01 = analyzer.calculate_dv01()
    print(f"\nDV01: ${dv01['dv01']:,.2f} per bp")
    print(f"Per $1M notional: ${dv01['dv01_per_million']:,.2f}")
    
    slope = analyzer.calculate_slope_risk()
    print(f"\nSlope Risk: ${slope['slope_risk']:,.2f}")
    print(f"Per bp steepening: ${slope['slope_risk_per_bp']:,.2f}")
    
    gamma = analyzer.calculate_gamma()
    print(f"\nGamma: {gamma['gamma']:,.2f}")
    print(f"Convexity: {gamma['convexity_pct']:.4f}%")
