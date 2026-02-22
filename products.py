import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from enum import Enum


class DayCountConvention(Enum):
    ACT_360 = "ACT/360"
    ACT_365 = "ACT/365"
    THIRTY_360 = "30/360"


@dataclass
class RangeAccrualProduct:
    #payoff = Notional * Coupon * (Days In Range / Total Days)
    
    notional: float
    coupon_rate: float  # Annual coupon rate (e.g., 0.05 for 5%)
    range_lower: float  # Lower bound of accrual range
    range_upper: float  # Upper bound of accrual range
    maturity: float     # Maturity in years
    cms_tenor: float    # CMS tenor in years (e.g., 10 for 10Y CMS)
    day_count: DayCountConvention = DayCountConvention.ACT_360
    
    def __post_init__(self):
        if self.notional <= 0:
            raise ValueError("Notional must be positive")
        if self.coupon_rate < 0:
            raise ValueError("Coupon rate must be non-negative")
        if self.range_lower >= self.range_upper:
            raise ValueError("Range lower must be less than range upper")
        if self.maturity <= 0:
            raise ValueError("Maturity must be positive")
        if self.cms_tenor <= 0:
            raise ValueError("CMS tenor must be positive")
    
    def calculate_accrual_fraction(self, cms_path: np.ndarray) -> float:
        #Calculate fraction of time CMS rate was in range.
        in_range = (cms_path >= self.range_lower) & (cms_path <= self.range_upper)
        return np.mean(in_range)
    
    def calculate_payoff(self, cms_path: np.ndarray) -> float:
        #Calculate product payoff for a single path. 
        #Args: cms_path: Array of CMS rates over the product life

        accrual_fraction = self.calculate_accrual_fraction(cms_path)
        
        # Adjust for day count convention
        if self.day_count == DayCountConvention.ACT_360:
            day_count_factor = 360 / 365
        elif self.day_count == DayCountConvention.ACT_365:
            day_count_factor = 1.0
        else:  # 30/360
            day_count_factor = 360 / 365
        
        payoff = (
            self.notional 
            * self.coupon_rate 
            * self.maturity 
            * accrual_fraction
            * day_count_factor
        )
        
        return payoff
    
    def payoff_function(self):
        def _payoff(paths: np.ndarray) -> np.ndarray:
            n_paths = paths.shape[0]
            payoffs = np.zeros(n_paths)
            
            for i in range(n_paths):
                payoffs[i] = self.calculate_payoff(paths[i, :])
            
            return payoffs
        
        return _payoff
    
    def __repr__(self) -> str:
        return (
            f"RangeAccrual(Notional={self.notional:,.0f}, "
            f"Coupon={self.coupon_rate:.2%}, "
            f"Range=[{self.range_lower:.2%}, {self.range_upper:.2%}], "
            f"Maturity={self.maturity}Y, CMS={self.cms_tenor}Y)"
        )


@dataclass
class DigitalRangeAccrual(RangeAccrualProduct):
    #Digital variant: pays full coupon if in range, nothing otherwise.
    
    #Payoff = Notional * Coupon * Maturity if always in range, else 0

    def calculate_payoff(self, cms_path: np.ndarray) -> float:

        accrual_fraction = self.calculate_accrual_fraction(cms_path)
        
        if accrual_fraction >= 0.999:
            day_count_factor = 360 / 365 if self.day_count == DayCountConvention.ACT_360 else 1.0
            return self.notional * self.coupon_rate * self.maturity * day_count_factor
        else:
            return 0.0


@dataclass
class KnockOutRangeAccrual(RangeAccrualProduct):
    # Knock-out variant: terminates early if rate breaches barrier.
    
    # CMS ever goes outside range, product knocks out with zero payoff.
    
    def calculate_payoff(self, cms_path: np.ndarray) -> float:

        # Check if ever breached
        breached = np.any(
            (cms_path < self.range_lower) | (cms_path > self.range_upper)
        )
        
        if breached:
            return 0.0
        else:
            # Standard accrual calculation
            return super().calculate_payoff(cms_path)


class ProductFactory:
    
    @staticmethod
    def create_standard_range_accrual(
        notional: float = 1_000_000,
        coupon_rate: float = 0.05,
        range_lower: float = 0.02,
        range_upper: float = 0.04,
        maturity: float = 5.0,
        cms_tenor: float = 10.0
    ) -> RangeAccrualProduct:
        return RangeAccrualProduct(
            notional=notional,
            coupon_rate=coupon_rate,
            range_lower=range_lower,
            range_upper=range_upper,
            maturity=maturity,
            cms_tenor=cms_tenor
        )
    
    @staticmethod
    def create_tight_range_product(
        notional: float = 1_000_000,
        current_rate: float = 0.03,
        range_width: float = 0.01,
        coupon_rate: float = 0.08,
        maturity: float = 3.0
    ) -> RangeAccrualProduct:
        return RangeAccrualProduct(
            notional=notional,
            coupon_rate=coupon_rate,
            range_lower=current_rate - range_width / 2,
            range_upper=current_rate + range_width / 2,
            maturity=maturity,
            cms_tenor=10.0
        )
    
    @staticmethod
    def create_digital_product(
        notional: float = 1_000_000,
        range_lower: float = 0.025,
        range_upper: float = 0.045,
        coupon_rate: float = 0.10,
        maturity: float = 2.0
    ) -> DigitalRangeAccrual:
        return DigitalRangeAccrual(
            notional=notional,
            coupon_rate=coupon_rate,
            range_lower=range_lower,
            range_upper=range_upper,
            maturity=maturity,
            cms_tenor=10.0
        )


if __name__ == "__main__":
    product = ProductFactory.create_standard_range_accrual(
        notional=1_000_000,
        coupon_rate=0.05,
        range_lower=0.025,
        range_upper=0.045
    )
    
    print(product)
    
    sample_path = np.linspace(0.03, 0.035, 100)
    accrual = product.calculate_accrual_fraction(sample_path)
    payoff = product.calculate_payoff(sample_path)
    
    print(f"\nSample path accrual: {accrual:.2%}")
    print(f"Sample payoff: ${payoff:,.2f}")
    
    digital = ProductFactory.create_digital_product()
    print(f"\n{digital}")
    
    in_range_path = np.ones(100) * 0.035
    print(f"Digital payoff (in range): ${digital.calculate_payoff(in_range_path):,.2f}")
    
    out_range_path = np.linspace(0.02, 0.05, 100)
    print(f"Digital payoff (out of range): ${digital.calculate_payoff(out_range_path):,.2f}")
