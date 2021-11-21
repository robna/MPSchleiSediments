from settings import Config
import statsmodels.formula.api as smf
import statsmodels.api as sm

def glm(data):
    """
    Build a generalised linear model (GLM) from a dataframe input according to the formula specified in settings.Config.
    """
    model = smf.glm(formula=Config.glm_formula, data=data, family=sm.families.Gamma())
    result = model.fit()

    print(result.summary())
    return result
