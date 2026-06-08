"""
阶段87: Agent时间序列预测系统

遵循"图难于其易，为大于其细"的原则，实现趋势分解、滑动平均、指数平滑、自回归预测。
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class TimeSeriesPoint:
    timestamp: float
    value: float
    label: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "value": self.value,
            "label": self.label,
        }


class TrendDecomposer:
    """趋势分解器"""

    def __init__(self, window_size: int = 3):
        self.window_size = window_size

    def moving_average(self, values: List[float]) -> List[float]:
        """计算滑动平均"""
        if not values:
            return []
        result = []
        half = self.window_size // 2
        for i in range(len(values)):
            start = max(0, i - half)
            end = min(len(values), i + half + 1)
            window = values[start:end]
            result.append(sum(window) / len(window))
        return result

    def detrend(self, values: List[float]) -> Tuple[List[float], List[float]]:
        """去趋势，返回 (趋势, 残差)"""
        trend = self.moving_average(values)
        residual = [v - t for v, t in zip(values, trend)]
        return trend, residual

    def linear_trend(self, values: List[float]) -> Tuple[float, float]:
        """线性回归拟合趋势: y = a*x + b"""
        n = len(values)
        if n == 0:
            return 0.0, 0.0
        x_mean = (n - 1) / 2
        y_mean = sum(values) / n
        numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        if denominator == 0:
            return 0.0, y_mean
        a = numerator / denominator
        b = y_mean - a * x_mean
        return a, b


class ExponentialSmoother:
    """指数平滑器"""

    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha

    def smooth(self, values: List[float]) -> List[float]:
        """一次指数平滑"""
        if not values:
            return []
        result = [values[0]]
        for i in range(1, len(values)):
            smoothed = self.alpha * values[i] + (1 - self.alpha) * result[-1]
            result.append(smoothed)
        return result

    def forecast(self, values: List[float], steps: int = 1) -> List[float]:
        """基于指数平滑预测未来值"""
        smoothed = self.smooth(values)
        last = smoothed[-1] if smoothed else 0.0
        return [last] * steps


class AutoRegressiveModel:
    """自回归模型（简化版）"""

    def __init__(self, order: int = 2):
        self.order = order
        self.coefficients: List[float] = []
        self._fitted = False

    def fit(self, values: List[float]) -> None:
        """拟合自回归系数（最小二乘法简化版）"""
        if len(values) <= self.order:
            self.coefficients = [1.0 / self.order] * self.order
            self._fitted = True
            return

        # 构建X和Y矩阵
        X = []
        Y = []
        for i in range(self.order, len(values)):
            X.append(values[i - self.order:i])
            Y.append(values[i])

        # 简化版：使用协方差估计
        n = len(X)
        if n == 0:
            self.coefficients = [0.0] * self.order
            self._fitted = True
            return

        # 计算各滞后项的相关系数作为系数
        coeffs = []
        for lag in range(1, self.order + 1):
            x_lag = [values[i - lag] for i in range(lag, len(values))]
            y_target = values[lag:]
            mean_x = sum(x_lag) / len(x_lag)
            mean_y = sum(y_target) / len(y_target)
            num = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_lag, y_target))
            den_x = sum((x - mean_x) ** 2 for x in x_lag)
            den_y = sum((y - mean_y) ** 2 for y in y_target)
            if den_x == 0 or den_y == 0:
                coeffs.append(0.0)
            else:
                corr = num / (den_x ** 0.5 * den_y ** 0.5)
                coeffs.append(corr)

        # 归一化
        total = sum(abs(c) for c in coeffs)
        if total > 0:
            coeffs = [c / total for c in coeffs]
        self.coefficients = coeffs
        self._fitted = True

    def predict(self, recent_values: List[float]) -> float:
        """基于最近值预测下一个值"""
        if not self._fitted or len(recent_values) < self.order:
            return sum(recent_values) / len(recent_values) if recent_values else 0.0
        recent = recent_values[-self.order:]
        return sum(c * v for c, v in zip(self.coefficients, recent))

    def forecast(self, values: List[float], steps: int = 1) -> List[float]:
        """多步预测"""
        if not self._fitted:
            self.fit(values)
        result = []
        current = list(values)
        for _ in range(steps):
            pred = self.predict(current)
            result.append(pred)
            current.append(pred)
        return result


class SeasonalityDetector:
    """季节性检测器"""

    def __init__(self, max_period: int = 12):
        self.max_period = max_period

    def detect_period(self, values: List[float]) -> int:
        """检测季节性周期（简化版自相关）"""
        if len(values) < 4:
            return 1
        best_period = 1
        best_score = -1.0
        for period in range(2, min(self.max_period + 1, len(values) // 2)):
            # 计算周期自相关
            correlations = []
            for i in range(period, len(values)):
                correlations.append(values[i] * values[i - period])
            score = sum(correlations) / len(correlations) if correlations else 0.0
            if score > best_score:
                best_score = score
                best_period = period
        return best_period

    def deseasonalize(self, values: List[float], period: int) -> Tuple[List[float], List[float]]:
        """去除季节性，返回 (去季节化值, 季节因子)"""
        if period <= 1 or len(values) < period * 2:
            return values, [1.0] * len(values)

        # 计算季节因子
        seasonal_factors = []
        for i in range(period):
            season_values = [values[j] for j in range(i, len(values), period)]
            seasonal_factors.append(sum(season_values) / len(season_values) if season_values else 0.0)

        # 归一化季节因子
        mean_factor = sum(seasonal_factors) / len(seasonal_factors)
        if mean_factor > 0:
            seasonal_factors = [f / mean_factor for f in seasonal_factors]

        # 去季节化
        deseasonalized = []
        for i, v in enumerate(values):
            factor = seasonal_factors[i % period]
            if factor != 0:
                deseasonalized.append(v / factor)
            else:
                deseasonalized.append(v)

        return deseasonalized, seasonal_factors


class TimeSeriesForecaster:
    """时间序列预测器"""

    def __init__(self):
        self.decomposer = TrendDecomposer()
        self.smoother = ExponentialSmoother()
        self.ar_model = AutoRegressiveModel()
        self.seasonality = SeasonalityDetector()

    def analyze(self, values: List[float]) -> Dict[str, Any]:
        """分析时间序列"""
        if not values:
            return {}

        # 趋势
        trend_ma = self.decomposer.moving_average(values)
        a, b = self.decomposer.linear_trend(values)

        # 季节性
        period = self.seasonality.detect_period(values)
        deseasonalized, seasonal_factors = self.seasonality.deseasonalize(values, period)

        # 平滑
        smoothed = self.smoother.smooth(values)

        return {
            "count": len(values),
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "trend_slope": a,
            "trend_intercept": b,
            "seasonal_period": period,
            "smoothed_last": smoothed[-1] if smoothed else 0.0,
        }

    def forecast(self, values: List[float], steps: int = 5,
                 method: str = "auto") -> List[float]:
        """预测未来值"""
        if not values:
            return [0.0] * steps

        if method == "auto":
            # 自动选择方法
            period = self.seasonality.detect_period(values)
            if period > 1 and len(values) >= period * 2:
                method = "seasonal_ar"
            else:
                method = "ar"

        if method == "ma":
            trend = self.decomposer.moving_average(values)
            last_trend = trend[-1] if trend else values[-1]
            return [last_trend] * steps

        elif method == "exponential":
            return self.smoother.forecast(values, steps)

        elif method == "ar":
            self.ar_model.fit(values)
            return self.ar_model.forecast(values, steps)

        elif method == "seasonal_ar":
            period = self.seasonality.detect_period(values)
            deseasonalized, factors = self.seasonality.deseasonalize(values, period)
            self.ar_model.fit(deseasonalized)
            preds = self.ar_model.forecast(deseasonalized, steps)
            # 重新加回季节性
            return [preds[i] * factors[(len(values) + i) % period]
                    for i in range(steps)]

        else:
            # 默认：使用最后一个值
            return [values[-1]] * steps

    def get_stats(self) -> Dict[str, Any]:
        return {
            "decomposer_window": self.decomposer.window_size,
            "smoother_alpha": self.smoother.alpha,
            "ar_order": self.ar_model.order,
            "max_seasonal_period": self.seasonality.max_period,
        }


def create_time_series_forecaster() -> TimeSeriesForecaster:
    return TimeSeriesForecaster()
