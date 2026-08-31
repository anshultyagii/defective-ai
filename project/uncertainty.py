def conformal_prediction(preds, residuals, alpha=0.1):
    # Dummy conformal prediction
    q = np.quantile(np.abs(residuals), 1 - alpha)
    return preds - q, preds + q
