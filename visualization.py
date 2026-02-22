

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from matplotlib.figure import Figure
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class Visualizer:
    
    @staticmethod
    def plot_yield_curve(
        curves: Dict[str, 'YieldCurve'],
        title: str = "Yield Curves",
        save_path: Optional[str] = None
    ) -> Figure:
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for name, curve in curves.items():
            ax.scatter(
                curve.tenors,
                curve.rates * 100,
                label=f"{name} (market)",
                s=50,
                alpha=0.6
            )
        
            tenors_fine = np.linspace(
                curve.tenors[0],
                curve.tenors[-1],
                200
            )
            rates_fine = [curve.get_rate(t) * 100 for t in tenors_fine]
            ax.plot(tenors_fine, rates_fine, label=name, linewidth=2)
        
        ax.set_xlabel("Tenor (years)", fontsize=12)
        ax.set_ylabel("Rate (%)", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_cms_paths(
        paths: np.ndarray,
        product: 'RangeAccrualProduct',
        n_display: int = 100,
        title: str = "CMS Rate Simulation Paths",
        save_path: Optional[str] = None
    ) -> Figure:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        n_paths, n_steps = paths.shape
        time_grid = np.linspace(0, product.maturity, n_steps)
        
        n_show = min(n_display, n_paths)
        for i in range(n_show):
            ax.plot(
                time_grid,
                paths[i, :] * 100,
                alpha=0.3,
                linewidth=0.5,
                color='blue'
            )
        mean_path = np.mean(paths, axis=0)
        ax.plot(
            time_grid,
            mean_path * 100,
            color='red',
            linewidth=2.5,
            label='Mean Path'
        )

        ax.axhline(
            y=product.range_lower * 100,
            color='green',
            linestyle='--',
            linewidth=2,
            label=f'Lower Bound ({product.range_lower:.2%})'
        )
        ax.axhline(
            y=product.range_upper * 100,
            color='green',
            linestyle='--',
            linewidth=2,
            label=f'Upper Bound ({product.range_upper:.2%})'
        )
        
        ax.fill_between(
            time_grid,
            product.range_lower * 100,
            product.range_upper * 100,
            alpha=0.2,
            color='green',
            label='Accrual Range'
        )
        
        ax.set_xlabel("Time (years)", fontsize=12)
        ax.set_ylabel("CMS Rate (%)", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_payoff_distribution(
        payoffs: np.ndarray,
        product: 'RangeAccrualProduct',
        title: str = "Payoff Distribution",
        save_path: Optional[str] = None
    ) -> Figure:

        fig, ax = plt.subplots(figsize=(12, 6))
        
        n, bins, patches = ax.hist(
            payoffs,
            bins=50,
            density=True,
            alpha=0.7,
            color='skyblue',
            edgecolor='black'
        )
    
        mean_payoff = np.mean(payoffs)
        median_payoff = np.median(payoffs)
        std_payoff = np.std(payoffs)
        
        ax.axvline(
            mean_payoff,
            color='red',
            linestyle='--',
            linewidth=2,
            label=f'Mean: ${mean_payoff:,.0f}'
        )
        ax.axvline(
            median_payoff,
            color='orange',
            linestyle='--',
            linewidth=2,
            label=f'Median: ${median_payoff:,.0f}'
        )
        
        stats_text = (
            f"Statistics:\n"
            f"Mean: ${mean_payoff:,.0f}\n"
            f"Std Dev: ${std_payoff:,.0f}\n"
            f"Min: ${np.min(payoffs):,.0f}\n"
            f"Max: ${np.max(payoffs):,.0f}"
        )
        ax.text(
            0.02, 0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
        
        ax.set_xlabel("Payoff ($)", fontsize=12)
        ax.set_ylabel("Probability Density", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_risk_ladder(
        ladder_risks: Dict[float, float],
        title: str = "Key Rate Durations (Ladder Risk)",
        save_path: Optional[str] = None
    ) -> Figure:
 
        fig, ax = plt.subplots(figsize=(12, 6))
        
        tenors = sorted(ladder_risks.keys())
        sensitivities = [ladder_risks[t] for t in tenors]
        
        colors = ['red' if s < 0 else 'green' for s in sensitivities]
        
        bars = ax.bar(
            [f"{t}Y" for t in tenors],
            sensitivities,
            color=colors,
            alpha=0.7,
            edgecolor='black'
        )
        
        for bar, sens in zip(bars, sensitivities):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                height,
                f'${sens:,.0f}',
                ha='center',
                va='bottom' if sens > 0 else 'top',
                fontsize=9
            )
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax.set_xlabel("Tenor", fontsize=12)
        ax.set_ylabel("Sensitivity ($)", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_scenario_pnl(
        scenario_results: Dict[str, Dict[str, float]],
        title: str = "Scenario P&L Analysis",
        save_path: Optional[str] = None
    ) -> Figure:

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        scenarios = list(scenario_results.keys())
        pnls = [scenario_results[s]['pnl'] for s in scenarios]
        pnl_pcts = [scenario_results[s]['pnl_pct'] for s in scenarios]
        
        colors = ['red' if p < 0 else 'green' for p in pnls]
        bars1 = ax1.bar(scenarios, pnls, color=colors, alpha=0.7, edgecolor='black')
        
        for bar, pnl in zip(bars1, pnls):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.,
                height,
                f'${pnl:,.0f}',
                ha='center',
                va='bottom' if pnl > 0 else 'top',
                fontsize=10
            )
        
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax1.set_xlabel("Scenario", fontsize=12)
        ax1.set_ylabel("P&L ($)", fontsize=12)
        ax1.set_title("P&L Impact", fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.tick_params(axis='x', rotation=45)

        colors2 = ['red' if p < 0 else 'green' for p in pnl_pcts]
        bars2 = ax2.bar(scenarios, pnl_pcts, color=colors2, alpha=0.7, edgecolor='black')
        
        for bar, pct in zip(bars2, pnl_pcts):
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.,
                height,
                f'{pct:+.2f}%',
                ha='center',
                va='bottom' if pct > 0 else 'top',
                fontsize=10
            )
        
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax2.set_xlabel("Scenario", fontsize=12)
        ax2.set_ylabel("P&L (%)", fontsize=12)
        ax2.set_title("P&L Impact (%)", fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.tick_params(axis='x', rotation=45)
        
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_sensitivity_surface(
        rate_shifts: np.ndarray,
        vol_changes: np.ndarray,
        prices: np.ndarray,
        title: str = "Price Sensitivity Surface",
        save_path: Optional[str] = None
    ) -> Figure:

        fig = plt.figure(figsize=(14, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        X, Y = np.meshgrid(rate_shifts * 10000, vol_changes * 100)  # Convert to bp and %
        
        surf = ax.plot_surface(
            X, Y, prices,
            cmap='viridis',
            alpha=0.8,
            edgecolor='none'
        )
        
        ax.set_xlabel("Rate Shift (bp)", fontsize=11)
        ax.set_ylabel("Vol Change (%)", fontsize=11)
        ax.set_zlabel("Price ($)", fontsize=11)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Price ($)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def create_risk_dashboard(
        product: 'RangeAccrualProduct',
        mc_results: dict,
        risk_metrics: 'RiskMetrics',
        save_path: Optional[str] = None
    ) -> Figure:
        
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        ax1 = fig.add_subplot(gs[0, 0])
        payoffs = mc_results['payoffs']
        ax1.hist(payoffs, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(np.mean(payoffs), color='red', linestyle='--', linewidth=2, label='Mean')
        ax1.set_xlabel("Payoff ($)")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Payoff Distribution", fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2 = fig.add_subplot(gs[0, 1])
        paths = mc_results['paths']
        time_grid = np.linspace(0, product.maturity, paths.shape[1])
        for i in range(min(50, paths.shape[0])):
            ax2.plot(time_grid, paths[i, :] * 100, alpha=0.3, linewidth=0.5)
        ax2.axhline(product.range_lower * 100, color='green', linestyle='--', linewidth=2)
        ax2.axhline(product.range_upper * 100, color='green', linestyle='--', linewidth=2)
        ax2.fill_between(
            time_grid,
            product.range_lower * 100,
            product.range_upper * 100,
            alpha=0.2,
            color='green'
        )
        ax2.set_xlabel("Time (years)")
        ax2.set_ylabel("CMS Rate (%)")
        ax2.set_title("Sample CMS Paths", fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.axis('off')
        
        risk_text = (
            f"PRODUCT DETAILS\n"
            f"{'=' * 40}\n"
            f"Notional: ${product.notional:,.0f}\n"
            f"Coupon: {product.coupon_rate:.2%}\n"
            f"Range: [{product.range_lower:.2%}, {product.range_upper:.2%}]\n"
            f"Maturity: {product.maturity} years\n\n"
            f"PRICING & RISK\n"
            f"{'=' * 40}\n"
            f"Price: ${mc_results['price']:,.2f}\n"
            f"Std Error: ${mc_results['std_error']:,.2f}\n"
            f"DV01: ${risk_metrics.dv01:,.2f} per bp\n"
            f"Slope Risk: ${risk_metrics.slope_risk:,.2f}\n"
            f"Gamma: {risk_metrics.gamma:,.2f}\n\n"
            f"STATISTICS\n"
            f"{'=' * 40}\n"
            f"Avg Payoff: ${mc_results['avg_payoff']:,.2f}\n"
            f"Payoff Std: ${mc_results['payoff_std']:,.2f}\n"
            f"Simulations: {mc_results['n_simulations']:,}"
        )
        
        ax3.text(
            0.1, 0.5,
            risk_text,
            transform=ax3.transAxes,
            fontsize=10,
            verticalalignment='center',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3)
        )
        
        ax4 = fig.add_subplot(gs[1, 1])
        
        in_range_probs = []
        for path in paths:
            in_range = np.mean(
                (path >= product.range_lower) & (path <= product.range_upper)
            )
            in_range_probs.append(in_range)
        
        ax4.hist(in_range_probs, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        ax4.axvline(np.mean(in_range_probs), color='red', linestyle='--', linewidth=2, label='Mean')
        ax4.set_xlabel("Fraction of Time in Range")
        ax4.set_ylabel("Frequency")
        ax4.set_title("Accrual Efficiency Distribution", fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        fig.suptitle(
            "CMS Range Accrual Risk Dashboard",
            fontsize=16,
            fontweight='bold',
            y=0.98
        )
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


if __name__ == "__main__":
    print("Visualization module loaded successfully")
    print("Available visualizations:")
    print("  - plot_yield_curve()")
    print("  - plot_cms_paths()")
    print("  - plot_payoff_distribution()")
    print("  - plot_risk_ladder()")
    print("  - plot_scenario_pnl()")
    print("  - create_risk_dashboard()")
