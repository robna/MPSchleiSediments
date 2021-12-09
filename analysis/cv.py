from statsmodels.sandbox.tools.cross_val import LeaveOneOut, split
import glm


def llo_cv(df):
    loo = LeaveOneOut(len(df))
    pred = []
    for train_index, test_index in loo:
        # this could be used for doing loo-cross-val, if using array-based instead formula-and-dataframe based model:
        #X_train, X_test, y_train, y_test = cross_val.split(train_index, test_index, X, y)  

        current_input = df.loc[train_index]
        glm_res = glm.glm(current_input)
        pred = pred.append(glm_res.predict(glm_input.loc[test_index]))
        
        
    # TODO: get error from prediction
    # RMSE = sum((yhat-y)**2 / n)