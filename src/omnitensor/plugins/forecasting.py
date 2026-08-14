"""Compatibility facade for neutral forecasting mechanics."""

from .. import forecasting as _forecasting

DEFAULT_RIDGE = _forecasting.DEFAULT_RIDGE
MAX_FEATURES_PER_FIT = _forecasting.MAX_FEATURES_PER_FIT
MIN_TRAINING_ROWS = _forecasting.MIN_TRAINING_ROWS
ForecastError = _forecasting.ForecastError
ForecastQuality = _forecasting.ForecastQuality
LinearForecaster = _forecasting.LinearForecaster
evaluate_forecast_predictions = _forecasting.evaluate_forecast_predictions
fit_forecaster = _forecasting.fit_forecaster
