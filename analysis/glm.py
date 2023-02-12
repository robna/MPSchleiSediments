from settings import Config
import statsmodels.formula.api as smf
import statsmodels.api as sm
import numpy as np

def glm(data):
    """
    Build a generalised linear model (GLM) from a dataframe input according to the formula specified in settings.Config.
    """
    power_power = f'{Config.glm_power_power}' if Config.glm_link == 'Power' else ''
    link = f'sm.families.links.{Config.glm_link}({power_power}), ' if Config.glm_link is not None else ''
    var_power = f'var_power={Config.glm_tweedie_power}' if Config.glm_family == 'Tweedie' else ''
    print('GLM link: ', link if Config.glm_link is not None else 'default')
    family = eval(f'sm.families.{Config.glm_family}({link}{var_power})')
    print('GLM family: ', family)
    model = smf.glm(formula=Config.glm_formula,
                    data=data,
                    family=family)
    u, s, vt = np.linalg.svd(model.exog, 0)
    print('Singular? All values should be positive: ', s)
    result = model.fit(method='lbfgs')

    return result
