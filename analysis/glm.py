from settings import Config
import statsmodels.formula.api as smf
import statsmodels.api as sm

def glm(data):
    """
    Build a generalised linear model (GLM) from a dataframe input according to the formula specified in settings.Config.
    """
    model = smf.glm(formula=Config.glm_formula, data=data, family=eval(f'sm.families.{Config.glm_family}()'))  # TODO: family is coded here, should be in settings > Config
    result = model.fit()

    return result
