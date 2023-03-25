from flask import Flask, render_template, request

from flask import Flask, render_template, request

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Load the dataset from a CSV file
        df = pd.read_csv(request.files['file'])

        # Get the name of the target variable from the user input
        target_name = request.form['target']                
        df = pd.DataFrame(df)
        #df.to_csv('df_17Mar.csv')
        df = df.replace(r'^\s*$', np.nan, regex=True)

        df = df[df[target_name].notna()]

        d_typ = pd.DataFrame(df.dtypes).reset_index()
        d_typ.rename(columns={'index': 'feature_name',
                    0: 'data_type'}, inplace=True)

        d_label_cnt = pd.DataFrame(df.nunique()).reset_index()
        d_label_cnt.rename(
            columns={'index': 'feature_name', 0: 'unique_cnt'}, inplace=True)
        d_typ_label = pd.merge(d_typ, d_label_cnt, on='feature_name', how='left')

        max_label_cut_off = 50
        d_typ_label['max_label_flag'] = np.where(
            d_typ_label['unique_cnt'] <= 50, 0, 1)

        var_to_drop = []
        for index, row in d_typ_label.iterrows():
            if (row['data_type'] == 'object') & (row['max_label_flag'] == 1):
                var_to_drop.append(row['feature_name'])

        if len(var_to_drop) != 0:
            df = df.drop(var_to_drop, axis=1)

        num_cols = list(df.select_dtypes(include='number').columns)

        if target_name in num_cols:
            num_cols.remove(target_name)

        cat_cols = list(set(df.columns) - set(num_cols))

        d_null = pd.DataFrame(df.isnull().sum()).reset_index()
        d_null.rename(columns={'index': 'feature_name',
                    0: 'missing_cnt'}, inplace=True)

        var_with_null = []
        for index, row in d_null.iterrows():
            if row['missing_cnt'] != 0:
                var_with_null.append(row['feature_name'])

        for var in var_with_null:
            if var in num_cols:
                df[var] = df[var].fillna(df[var].median())
            else:
                df[var] = df[var].fillna(df[var].mode()[0])

        df_2 = pd.get_dummies(df, drop_first=True)

        st_trans = StandardScaler()
        df_2[num_cols] = st_trans.fit_transform(df_2[num_cols])

        X = df_2.drop(target_name, axis=1)
        y = df_2[target_name]
    ################################################

    ##########################################################################################
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=1663)

        print("Fit Logistic Regression !!")
        lr = LogisticRegression(max_iter=10000)
        lr.fit(X_train, y_train)

        y_pred_lr = lr.predict(X_test)
        accuracy_lr = accuracy_score(y_test, y_pred_lr)
        precision_lr = precision_score(y_test, y_pred_lr)
        recall_lr = recall_score(y_test, y_pred_lr)
        f1_score_lr = 2*(precision_lr * recall_lr) / (precision_lr + recall_lr)

        rf = RandomForestClassifier()
        rf.fit(X_train, y_train)

        y_pred_rf = rf.predict(X_test)
        accuracy_rf = accuracy_score(y_test, y_pred_rf)
        precision_rf = precision_score(y_test, y_pred_rf)
        recall_rf = recall_score(y_test, y_pred_rf)
        f1_score_rf = 2*(precision_rf * recall_rf) / (precision_rf + recall_rf)

        gbm = GradientBoostingClassifier()
        gbm.fit(X_train, y_train)

        y_pred_gbm = gbm.predict(X_test)
        accuracy_gbm = accuracy_score(y_test, y_pred_gbm)
        precision_gbm = precision_score(y_test, y_pred_gbm)
        recall_gbm = recall_score(y_test, y_pred_gbm)
        f1_score_gbm = 2*(precision_gbm * recall_gbm) / \
            (precision_gbm + recall_gbm)

        xgb = XGBClassifier()
        xgb.fit(X_train, y_train)

        y_pred_xgb = xgb.predict(X_test)
        accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
        precision_xgb = precision_score(y_test, y_pred_xgb)
        recall_xgb = recall_score(y_test, y_pred_xgb)
        f1_score_xgb = 2*(precision_xgb * recall_xgb) / \
            (precision_xgb + recall_xgb)

        svc = SVC(gamma='auto')
        svc.fit(X_train, y_train)

        y_pred_svc = svc.predict(X_test)
        accuracy_svc = accuracy_score(y_test, y_pred_svc)
        precision_svc = precision_score(y_test, y_pred_svc)
        recall_svc = recall_score(y_test, y_pred_svc)
        f1_score_svc = 2*(precision_svc * recall_svc) / \
            (precision_svc + recall_svc)

        accuracy_dict = {'LogisticRegression': accuracy_lr,
                        'RandomForest': accuracy_rf,
                        'GBM': accuracy_gbm,
                        'XGBoost': accuracy_xgb,
                        'SupportVectorMachine': accuracy_svc}

        precision_dict = {'LogisticRegression': precision_lr,
                        'RandomForest': precision_rf,
                        'GBM': precision_gbm,
                        'XGBoost': precision_xgb,
                        'SupportVectorMachine': precision_svc}

        recall_dict = {'LogisticRegression': recall_lr,
                    'RandomForest': recall_rf,
                    'GBM': recall_gbm,
                    'XGBoost': recall_xgb,
                    'SupportVectorMachine': recall_svc}

        fi_score_dict = {'LogisticRegression': f1_score_lr,
                        'RandomForest': f1_score_rf,
                        'GBM': f1_score_gbm,
                        'XGBoost': f1_score_xgb,
                        'SupportVectorMachine': f1_score_svc}

        # To find best model we will compare F1 score
        f1_score_list = [f1_score_rf, f1_score_gbm, f1_score_xgb, f1_score_svc]
        best_f1_score = f1_score_lr
        for i in f1_score_list:
            if i > best_f1_score:
                best_f1_score = i

        #get best model name having best F1 score amoung 5 models
        best_model = list(fi_score_dict.keys())[list(
            fi_score_dict.values()).index(best_f1_score)]

        best_model = best_model
        best_model_f1_score = best_f1_score
        print("best_model : ", best_model)
        print("best_model_f1_score : ", best_f1_score)

    #  return best_model, best_model_f1_score

    # Save the model to a file
       # jb.dump(best_model, 'model.pkl')

        # Render the result template with the model score
        #return render_template('result.html', score=best_model_f1_score)
        return render_template('result.html', score=best_model_f1_score, model=best_model)
  

    except Exception as e:
        # Render the error template with the error message
        return render_template('error.html', error=str(e))


if __name__ == '__main__':
    app.run(debug=True)
