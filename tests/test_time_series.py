"""阶段87: Agent时间序列预测系统 - 单元测试"""
import pytest
from src.agent.time_series import (
    TimeSeriesPoint,
    TrendDecomposer, ExponentialSmoother, AutoRegressiveModel,
    SeasonalityDetector, TimeSeriesForecaster, create_time_series_forecaster,
)


class TestTrendDecomposer:
    def test_moving_average(self):
        d = TrendDecomposer(window_size=3)
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        ma = d.moving_average(values)
        assert len(ma) == 5
        assert ma[2] == 3.0  # 中间点

    def test_detrend(self):
        d = TrendDecomposer()
        values = [1.0, 2.0, 3.0]
        trend, residual = d.detrend(values)
        assert len(trend) == 3
        assert len(residual) == 3

    def test_linear_trend(self):
        d = TrendDecomposer()
        values = [0.0, 1.0, 2.0, 3.0, 4.0]
        a, b = d.linear_trend(values)
        assert abs(a - 1.0) < 0.001  # 斜率应为1

    def test_empty(self):
        d = TrendDecomposer()
        assert d.moving_average([]) == []
        assert d.linear_trend([]) == (0.0, 0.0)


class TestExponentialSmoother:
    def test_smooth(self):
        s = ExponentialSmoother(alpha=0.5)
        values = [10.0, 12.0, 14.0]
        smoothed = s.smooth(values)
        assert smoothed[0] == 10.0
        assert smoothed[1] == 11.0  # 0.5*12 + 0.5*10

    def test_forecast(self):
        s = ExponentialSmoother(alpha=0.5)
        values = [10.0, 12.0, 14.0]
        forecast = s.forecast(values, steps=2)
        assert len(forecast) == 2
        assert forecast[0] == forecast[1]  # 指数平滑预测值相同


class TestAutoRegressiveModel:
    def test_fit(self):
        ar = AutoRegressiveModel(order=2)
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        ar.fit(values)
        assert ar._fitted
        assert len(ar.coefficients) == 2

    def test_predict(self):
        ar = AutoRegressiveModel(order=2)
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        ar.fit(values)
        pred = ar.predict(values)
        assert isinstance(pred, float)

    def test_forecast(self):
        ar = AutoRegressiveModel(order=2)
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        preds = ar.forecast(values, steps=3)
        assert len(preds) == 3


class TestSeasonalityDetector:
    def test_detect_period_no_seasonality(self):
        s = SeasonalityDetector()
        values = [1.0, 2.0, 3.0, 4.0]
        period = s.detect_period(values)
        assert isinstance(period, int)

    def test_detect_period_with_seasonality(self):
        s = SeasonalityDetector()
        # 周期为3的数据
        values = [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
        period = s.detect_period(values)
        assert period == 3

    def test_deseasonalize(self):
        s = SeasonalityDetector()
        values = [2.0, 4.0, 6.0, 2.0, 4.0, 6.0]
        deseasonalized, factors = s.deseasonalize(values, period=3)
        assert len(deseasonalized) == 6
        assert len(factors) == 3

    def test_deseasonalize_short(self):
        s = SeasonalityDetector()
        values = [1.0, 2.0]
        deseasonalized, factors = s.deseasonalize(values, period=3)
        assert deseasonalized == values


class TestTimeSeriesForecaster:
    def test_analyze(self):
        f = TimeSeriesForecaster()
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        analysis = f.analyze(values)
        assert analysis["count"] == 5
        assert analysis["mean"] == 3.0
        assert "trend_slope" in analysis

    def test_forecast_ma(self):
        f = TimeSeriesForecaster()
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        preds = f.forecast(values, steps=2, method="ma")
        assert len(preds) == 2

    def test_forecast_exponential(self):
        f = TimeSeriesForecaster()
        values = [10.0, 12.0, 14.0]
        preds = f.forecast(values, steps=2, method="exponential")
        assert len(preds) == 2

    def test_forecast_ar(self):
        f = TimeSeriesForecaster()
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        preds = f.forecast(values, steps=2, method="ar")
        assert len(preds) == 2

    def test_forecast_auto(self):
        f = TimeSeriesForecaster()
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        preds = f.forecast(values, steps=2, method="auto")
        assert len(preds) == 2

    def test_forecast_seasonal(self):
        f = TimeSeriesForecaster()
        values = [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
        preds = f.forecast(values, steps=3, method="seasonal_ar")
        assert len(preds) == 3

    def test_forecast_empty(self):
        f = TimeSeriesForecaster()
        preds = f.forecast([], steps=3)
        assert preds == [0.0, 0.0, 0.0]

    def test_get_stats(self):
        f = TimeSeriesForecaster()
        stats = f.get_stats()
        assert "decomposer_window" in stats
        assert "smoother_alpha" in stats
        assert "ar_order" in stats

    def test_create_time_series_forecaster(self):
        f = create_time_series_forecaster()
        assert isinstance(f, TimeSeriesForecaster)
