from settings import Config
import statsmodels.formula.api as smf
import statsmodels.api as sm

def glm(data):
    """
    Build a generalised linear model (GLM) from a dataframe input according to the formula specified in settings.Config.
    """
    link = f'sm.families.links.{Config.glm_link}()' if Config.glm_link is not None else ''
    print(link)
    family = eval(f'sm.families.{Config.glm_family}({link})')
    print(family)
    
    model = smf.glm(formula=Config.glm_formula,
                    data=data,
                    family=family)
    result = model.fit()

    return result
