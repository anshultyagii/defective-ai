import shap
import matplotlib.pyplot as plt
def explain_model(model, X):
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        shap.summary_plot(shap_values, X, show=False)
        plt.savefig('shap_summary.png')
        plt.close()
    except Exception as e:
        print(f"SHAP explainer failed (likely unsupported model type): {e}")
