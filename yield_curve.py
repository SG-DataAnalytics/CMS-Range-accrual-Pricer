import numpy as np
from scipy.interpolate import CubicSpline, interp1d
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum


class InterpolationMethod(Enum):
    LINEAR = "linear"
    CUBIC_SPLINE = "cubic_spline"
    LOG_LINEAR = "log_linear"


@dataclass
class MarketQuote:
    tenor: float  
    rate: float  
    
    def __post_init__(self):
        if self.tenor <= 0:
            raise ValueError(f"Tenor must be positive, got {self.tenor}")
        if not -0.1 <= self.rate <= 0.5:
            raise ValueError(f"Rate seems unrealistic: {self.rate}")


class YieldCurve:
    
    def __init__(
        self,
        quotes: List[MarketQuote],
        curve_name: str = "Curve",
        interpolation: InterpolationMethod = InterpolationMethod.CUBIC_SPLINE
    ):
        self.curve_name = curve_name
        self.interpolation = interpolation
        
        sorted_quotes = sorted(quotes, key=lambda q: q.tenor)
        self.tenors = np.array([q.tenor for q in sorted_quotes])
        self.rates = np.array([q.rate for q in sorted_quotes])
        
        if len(self.tenors) < 2:
            raise ValueError("Need at least 2 quotes to build a curve")
        
        self._interpolator = self._build_interpolator()
    
    def _build_interpolator(self):
        if self.interpolation == InterpolationMethod.LINEAR:
            return interp1d(
                self.tenors, 
                self.rates, 
                kind='linear',
                fill_value='extrapolate'
            )
        elif self.interpolation == InterpolationMethod.CUBIC_SPLINE:
            return CubicSpline(
                self.tenors, 
                self.rates,
                extrapolate=True
            )
        elif self.interpolation == InterpolationMethod.LOG_LINEAR:
            # Log-linear on discount factors
            dfs = np.exp(-self.rates * self.tenors)
            log_dfs = np.log(dfs)
            return interp1d(
                self.tenors,
                log_dfs,
                kind='linear',
                fill_value='extrapolate'
            )
        else:
            raise ValueError(f"Unknown interpolation: {self.interpolation}")
    
    def get_rate(self, tenor: float) -> float:
        #Interpolation
        if tenor < 0:
            raise ValueError(f"Tenor must be non-negative, got {tenor}")
        
        if self.interpolation == InterpolationMethod.LOG_LINEAR:
            log_df = float(self._interpolator(tenor))
            df = np.exp(log_df)
            return -np.log(df) / tenor if tenor > 0 else self.rates[0]
        else:
            return float(self._interpolator(tenor))
    
    def get_discount_factor(self, tenor: float) -> float:
        if tenor == 0:
            return 1.0
        
        rate = self.get_rate(tenor)
        return np.exp(-rate * tenor)
    
    def get_forward_rate(self, t1: float, t2: float) -> float:
        if t2 <= t1:
            raise ValueError(f"t2 ({t2}) must be greater than t1 ({t1})")
        
        df1 = self.get_discount_factor(t1)
        df2 = self.get_discount_factor(t2)
        
        return (df1 / df2 - 1) / (t2 - t1)
    
    def bump_curve(self, bump_size: float) -> 'YieldCurve':
        #shifted rate curve
        bumped_quotes = [
            MarketQuote(tenor=t, rate=r + bump_size)
            for t, r in zip(self.tenors, self.rates)
        ]
        return YieldCurve(
            quotes=bumped_quotes,
            curve_name=f"{self.curve_name}_bumped",
            interpolation=self.interpolation
        )
    
    def __repr__(self) -> str:
        return f"YieldCurve(name='{self.curve_name}', points={len(self.tenors)})"


class CurveBuilder:
    
    @staticmethod
    def build_ois_curve(
        market_data: Dict[float, float],
        interpolation: InterpolationMethod = InterpolationMethod.CUBIC_SPLINE
    ) -> YieldCurve:

        quotes = [MarketQuote(tenor=t, rate=r) for t, r in market_data.items()]
        return YieldCurve(quotes, curve_name="OIS", interpolation=interpolation)
    
    @staticmethod
    def build_swap_curve(
        market_data: Dict[float, float],
        interpolation: InterpolationMethod = InterpolationMethod.CUBIC_SPLINE
    ) -> YieldCurve:
 
        quotes = [MarketQuote(tenor=t, rate=r) for t, r in market_data.items()]
        return YieldCurve(quotes, curve_name="Swap", interpolation=interpolation)
    
    @staticmethod
    def build_sample_curves() -> Tuple[YieldCurve, YieldCurve]:

        # Sample OIS rates
        ois_data = {
            0.25: 0.0200,
            0.50: 0.0215,
            1.0:  0.0235,
            2.0:  0.0265,
            3.0:  0.0285,
            5.0:  0.0310,
            7.0:  0.0325,
            10.0: 0.0340,
            15.0: 0.0350,
            20.0: 0.0355,
        }
        
        # Sample swap rates
        swap_data = {
            0.25: 0.0210,
            0.50: 0.0225,
            1.0:  0.0245,
            2.0:  0.0275,
            3.0:  0.0295,
            5.0:  0.0320,
            7.0:  0.0335,
            10.0: 0.0350,
            15.0: 0.0360,
            20.0: 0.0365,
        }
        
        ois_curve = CurveBuilder.build_ois_curve(ois_data)
        swap_curve = CurveBuilder.build_swap_curve(swap_data)
        
        return ois_curve, swap_curve


if __name__ == "__main__":
    ois_curve, swap_curve = CurveBuilder.build_sample_curves()
    
    print(f"\n{ois_curve}")
    print(f"5Y OIS rate: {ois_curve.get_rate(5.0):.4%}")
    print(f"5Y discount factor: {ois_curve.get_discount_factor(5.0):.6f}")
    
    print(f"\n{swap_curve}")
    print(f"5Y swap rate: {swap_curve.get_rate(5.0):.4%}")
    print(f"2Y-5Y forward rate: {swap_curve.get_forward_rate(2.0, 5.0):.4%}")
