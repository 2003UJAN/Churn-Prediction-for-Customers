import shap
import matplotlib.pyplot as plt

def explain_model(model, X_train):
    explainer = shap.Explainer(model)
    shap_values = explainer(X_train)

    shap.summary_plot(shap_values, X_train, show=False)
    plt.tight_layout()
    plt.savefig("shap_summary.png", bbox_inches='tight')
