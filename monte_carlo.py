import numpy as np
from typing import Optional, Callable, List
from dataclasses import dataclass
from yield_curve import YieldCurve


@dataclass
class ModelParameters:
    mean_reversion: float = 0.05  # Speed of mean reversion
    volatility: float = 0.015      # Annual volatility
    long_term_mean: float = 0.03   # Long-term mean rate
    
    def __post_init__(self):
        if self.mean_reversion < 0:
            raise ValueError("Mean reversion must be non-negative")
        if self.volatility <= 0:
            raise ValueError("Volatility must be positive")


class CMSRateSimulator:
    """
    mean-reverting process (Vasicek-type) for simplicity and speed:
    dr = alpha * (mu - r) * dt + sigma * dW
    """
    
    def __init__(
        self,
        initial_rate: float,
        parameters: ModelParameters,
        random_seed: Optional[int] = None
    ):
        self.initial_rate = initial_rate
        self.params = parameters
        
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def simulate_paths(
        self,
        n_paths: int,
        n_steps: int,
        time_horizon: float
    ) -> np.ndarray:

        dt = time_horizon / n_steps
        sqrt_dt = np.sqrt(dt)
        
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = self.initial_rate
        
        for i in range(n_steps):
            current_rate = paths[:, i]
            
            drift = self.params.mean_reversion * (
                self.params.long_term_mean - current_rate
            ) * dt
            
            diffusion = self.params.volatility * sqrt_dt * np.random.randn(n_paths)
            
            paths[:, i + 1] = np.maximum(current_rate + drift + diffusion, 0.0001)
        
        return paths
    
    def simulate_single_path(
        self,
        n_steps: int,
        time_horizon: float
    ) -> np.ndarray:
        return self.simulate_paths(1, n_steps, time_horizon)[0]


class MonteCarloEngine:
    def __init__(
        self,
        discount_curve: YieldCurve,
        n_simulations: int = 10000,
        random_seed: Optional[int] = 42
    ):
        self.discount_curve = discount_curve
        self.n_simulations = n_simulations
        self.random_seed = random_seed
    
    def price_product(
        self,
        simulator: CMSRateSimulator,
        payoff_function: Callable[[np.ndarray], np.ndarray],
        maturity: float,
        n_steps: int = 252
    ) -> dict:
       
        paths = simulator.simulate_paths(
            n_paths=self.n_simulations,
            n_steps=n_steps,
            time_horizon=maturity
        )
        
        payoffs = payoff_function(paths)
        
        discount_factor = self.discount_curve.get_discount_factor(maturity)
        discounted_payoffs = payoffs * discount_factor
        
        price = np.mean(discounted_payoffs)
        std_error = np.std(discounted_payoffs) / np.sqrt(self.n_simulations)
        
        return {
            'price': price,
            'std_error': std_error,
            'discount_factor': discount_factor,
            'avg_payoff': np.mean(payoffs),
            'payoff_std': np.std(payoffs),
            'n_simulations': self.n_simulations,
            'paths': paths,
            'payoffs': payoffs
        }
    
    def calculate_greeks(
        self,
        simulator: CMSRateSimulator,
        payoff_function: Callable[[np.ndarray], np.ndarray],
        maturity: float,
        bump_size: float = 0.0001,
        n_steps: int = 252
    ) -> dict:
        
        base_result = self.price_product(
            simulator, payoff_function, maturity, n_steps
        )
        base_price = base_result['price']
        
        bumped_curve = self.discount_curve.bump_curve(bump_size)
        bumped_engine = MonteCarloEngine(
            discount_curve=bumped_curve,
            n_simulations=self.n_simulations,
            random_seed=self.random_seed
        )
        
        bumped_result = bumped_engine.price_product(
            simulator, payoff_function, maturity, n_steps
        )
        bumped_price = bumped_result['price']
        
        dv01 = (bumped_price - base_price) / bump_size
        
        return {
            'base_price': base_price,
            'bumped_price': bumped_price,
            'dv01': dv01,
            'dv01_per_bp': dv01 * 10000,
            'bump_size': bump_size
        }
    
    def scenario_analysis(
        self,
        simulator: CMSRateSimulator,
        payoff_function: Callable[[np.ndarray], np.ndarray],
        maturity: float,
        curve_shifts: List[float],
        n_steps: int = 252
    ) -> dict:
        results = {}
        
        for shift in curve_shifts:
            if shift == 0:
                engine = self
            else:
                shifted_curve = self.discount_curve.bump_curve(shift)
                engine = MonteCarloEngine(
                    discount_curve=shifted_curve,
                    n_simulations=self.n_simulations,
                    random_seed=self.random_seed
                )
            
            result = engine.price_product(
                simulator, payoff_function, maturity, n_steps
            )
            
            results[shift] = {
                'price': result['price'],
                'std_error': result['std_error']
            }
        
        return results


if __name__ == "__main__":
    from yield_curve import CurveBuilder

    ois_curve, _ = CurveBuilder.build_sample_curves()
    
    params = ModelParameters(
        mean_reversion=0.05,
        volatility=0.015,
        long_term_mean=0.03
    )
    simulator = CMSRateSimulator(
        initial_rate=0.032,
        parameters=params,
        random_seed=42
    )
    
    paths = simulator.simulate_paths(n_paths=5, n_steps=100, time_horizon=5.0)
    print(f"Simulated {paths.shape[0]} paths with {paths.shape[1]} steps")
    print(f"Final rates: {paths[:, -1]}")
    
    mc_engine = MonteCarloEngine(discount_curve=ois_curve, n_simulations=1000)
    
    def fixed_payoff(paths):
        return np.ones(paths.shape[0]) * 0.01  # 1% fixed
    
    result = mc_engine.price_product(simulator, fixed_payoff, maturity=5.0)
    print(f"\nTest price: {result['price']:.6f} ± {result['std_error']:.6f}")
