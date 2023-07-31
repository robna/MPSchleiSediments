import pandas as pd
from collections import defaultdict

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, RegressorMixin
from sklearn.model_selection import ParameterGrid
from sklearn.utils.metaestimators import available_if

from statsmodels.formula.api import glm as glm_sm

from joblib import Parallel, delayed, cpu_count
import joblib
import contextlib
from itertools import chain


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """
    Context manager to patch joblib to report into tqdm progress bar given as argument
    Note: from https://stackoverflow.com/a/58936697
    in a script running a joblib.Parallel loop, call in via context manager:
        with tqdm_joblib(tqdm(desc="My calculation", total=10)) as progress_bar:
            Parallel(n_jobs=16)(delayed(sqrt)(i**2) for i in range(10))  # OBS: be sure to change the total value to the actual number of iterations
    """
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


class PipelineHelper(BaseEstimator, TransformerMixin, ClassifierMixin):
    """
    Source: https://github.com/bmurauer/pipelinehelper
    Original author: bmurauer
    License: GPLv3
    This class can be used in scikit pipelines to select elements.
    In addition to the "replace_estimator" functionality of scikit itself,
    this class allows to set specified parameters for each option in the list.
    """

    def __init__(
            self,
            available_models=None,
            selected_model=None,
            include_bypass=False,
            optional=False,
    ):
        """
        Selects elements from a list to use as estimators in a pipeline.
        Args:
            available_models: a list of models which should be selected from.
                If you have time on your hands, please enable the use of
                pipelines here.
            selected_model: this parameter is required for the clone operation
                used by gridsearch. It should only be used initially if no grid
                search is used.
            optional: if set to true, one of the resulting configurations will
                have this stage empty.
        """
        self.optional = optional
        # this is required for the clone operator used in gridsearch
        self.selected_model = selected_model

        # cloned
        if type(available_models) == dict:
            self.available_models = available_models
        else:
            # manually initialized
            self.available_models = {}
            for (key, model) in available_models:
                self.available_models[key] = model

    def generate(self, param_dict=None):
        """
        Generates the parameters that are required for a gridsearch.
        Args:
            param_dict: parameters for the available models provided in the
                constructor. Note that these don't require the prefix path of
                all elements higher up the hierarchy of this TransformerPicker.
        """
        if param_dict is None:
            param_dict = dict()
        per_model_parameters = defaultdict(lambda: defaultdict(list))

        # collect parameters for each specified model
        for k, values in param_dict.items():
            # example:  randomforest__n_estimators
            model_name = k.split('__')[0]
            param_name = k[len(model_name) + 2:]
            if model_name not in self.available_models:
                raise Exception('no such model: {0}'.format(model_name))
            per_model_parameters[model_name][param_name] = values

        ret = []

        # create instance for cartesion product of all available parameters
        # for each model
        for model_name, param_dict in per_model_parameters.items():
            parameter_sets = ParameterGrid(param_dict)
            for parameters in parameter_sets:
                ret.append((model_name, parameters))

        # for every model that has no specified parameters, add default value
        for model_name in self.available_models.keys():
            if model_name not in per_model_parameters:
                ret.append((model_name, dict()))

        if self.optional:
            ret.append((None, dict()))
        return ret

    def get_params(self, deep=True):
        """
        Returns the parameters of the current TransformerPicker instance.
        Note that this is different from the parameters used by the selected
        model. Provided for scikit estimator compatibility.
        """
        result = {
            'available_models': self.available_models,
            'selected_model': self.selected_model,
            'optional': self.optional,
        }
        if deep and self.selected_model:
            result.update({
                'selected_model__' + k: v
                for k, v in self.selected_model.get_params(deep=True).items()
            })
        if deep and self.available_models:
            for name, model in self.available_models.items():
                result['available_models__' + name] = model
                result.update({
                    'available_models__' + name + '__' + k: v
                    for k, v in model.get_params(deep=True).items()
                })
        return result

    @property
    def transformer_list(self):
        """
        Returns a list of all available models.
        Provided for scikit estimator compatibility.
        """
        return self.available_models

    def set_params(self,
                   selected_model,
                   available_models=None,
                   optional=False):
        """
        Sets the parameters to all available models.
        Provided for scikit estimator compatibility.
        """
        if available_models:
            self.available_models = available_models

        if selected_model[0] is None:
            self.selected_model = None
        else:
            if selected_model[0] not in self.available_models:
                raise ValueError(
                    'trying to set selected model {selected_model[0]}, which '
                    f'is not in the available models {available_models}.'
                )
            self.selected_model = self.available_models[selected_model[0]]
            if self.selected_model is not None:
                self.selected_model.set_params(**selected_model[1])
        return self

    def fit(self, x, y=None, **kwargs):
        """Fits the selected model."""
        if self.selected_model is None or self.selected_model == 'passthrough':
            return self
        else:
            return self.selected_model.fit(x, y, **kwargs)

    def transform(self, x, *args, **kwargs):
        """Transforms data with the selected model."""
        if self.selected_model is None or self.selected_model == 'passthrough':
            return x
        else:
            return self.selected_model.transform(x, *args, **kwargs)

    def predict(self, x):
        """Predicts data with the selected model."""
        if self.optional:
            raise ValueError('a classifier can not be optional')
        return self.selected_model.predict(x)

    @available_if('selected_model')
    def predict_proba(self, x):
        return self.selected_model.predict_proba(x)

    @available_if('selected_model')
    def decision_function(self, x):
        return self.selected_model.decision_function(x)

    @property
    def classes_(self):
        if hasattr(self.selected_model, 'classes_'):
            return self.selected_model.classes_
        raise ValueError('selected model does not provide classes_')


# This is an example wrapper for statsmodels GLM
class SMWrapper(BaseEstimator, RegressorMixin):
    """
    Wrapper for statsmodels GLM (used with formula API) to be incorporated in a sklearn pipeline.
    In addition to the sklearn interface, it has also the statsmodels function summary(),
    which gives the info about p-values, R2 and other statistics.
    Source: https://stackoverflow.com/a/59100482/10381546 and https://stackoverflow.com/a/60234758/10381546

    Example of use:

        cols = ['feature1','feature2']
        X_train = df_train[cols].values
        X_test = df_test[cols].values
        y_train = df_train['label']
        y_test = df_test['label']
        model = SMWrapper()
        model.fit(X_train, y_train)
        model.summary()
        model.predict(X_test)


    If you want to show the names of the columns, you can call:

        model.fit(X_train, y_train, column_names=cols)


    To use it in cross_validation:

        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(SMWrapper(), X_train, y_train, cv=10, scoring='neg_mean_squared_error')
        scores
    """

    def __init__(self, family, formula, alpha, L1_wt):
        self.family = family
        self.formula = formula
        self.alpha = alpha
        self.L1_wt = L1_wt
        self.model = None
        self.result = None
    
    def fit(self, X, y):
        data = pd.concat([pd.DataFrame(X), pd.Series(y)], axis=1)
        data.columns = X.columns.tolist() + ['y']
        self.model = glm_sm(self.formula, data, family=self.family)
        self.result = self.model.fit_regularized(alpha=self.alpha, L1_wt=self.L1_wt, refit=True)
        return self.result
    
    def predict(self, X):
        return self.result.predict(X)

    def get_params(self, deep = False):
        return {'fit_intercept':self.fit_intercept}

    def summary(self):
        print(self.results_.summary())


def parallel(func=None, args=(), merge_func=lambda x:x, parallelism = cpu_count()):
    """
    TODO: this is not tested!!
    Decorator to parallelize a function.
    Source: https://bytepawn.com/python-decorators-for-data-scientists.html
    
        Example of use:
    
            @parallel
            def my_function(x):
                return x**2
    
            my_function([1,2,3,4,5,6,7,8,9,10])
    """

    def decorator(func: lambda li: sorted(chain(*li))):
        def inner(*args, **kwargs):
            results = Parallel(n_jobs=parallelism)(delayed(func)(*args, **kwargs) for i in range(parallelism))
            return merge_func(results)
        return inner
    if func is None:
        # decorator was used like @parallel(...)
        return decorator
    else:
        # decorator was used like @parallel, without parens
        return decorator(func)